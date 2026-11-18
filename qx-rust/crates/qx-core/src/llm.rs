//! 本地大模型客户端。
//!
//! 原 Python 版用 urllib 调 Ollama `/api/chat`。Rust 版刻意不引入
//! tokio/reqwest 整条异步链：桌面应用里 LLM 调用本就应放在后台线程，
//! 一个阻塞式 HTTP/1.1 客户端即可，依赖更少、启动更快（优化点）。

use serde::{Deserialize, Serialize};
use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::time::Duration;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
    /// 随消息附带的图片 base64（不含 `data:` 前缀）。Ollama `/api/chat` 原生支持
    /// 该字段，qwen3.5:4b 自带视觉投影器，可直接看图。空数组不序列化，纯文本
    /// 请求体与迁移前完全一致。
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub images: Vec<String>,
}

impl Message {
    pub fn system(content: impl Into<String>) -> Self {
        Self { role: "system".into(), content: content.into(), images: Vec::new() }
    }
    pub fn user(content: impl Into<String>) -> Self {
        Self { role: "user".into(), content: content.into(), images: Vec::new() }
    }
    pub fn with_images(mut self, images: Vec<String>) -> Self {
        self.images = images;
        self
    }
}

pub trait LlmClient: Send + Sync {
    fn chat(
        &self,
        messages: &[Message],
        temperature: f32,
        num_predict: u32,
    ) -> Result<String, String>;
    fn available(&self) -> bool {
        true
    }
    /// 是否具备视觉输入能力。Ollama 的 qwen3.5:4b 带视觉投影器，默认 true；
    /// 纯文本 GGUF 与占位实现应覆盖为 false，让上层给出明确的降级话术。
    fn supports_vision(&self) -> bool {
        true
    }
}

/// 无大模型时的占位实现：断网/Ollama 未安装也能完整演示知识库（对齐原项目降级设计）。
pub struct NoLlm;

impl LlmClient for NoLlm {
    fn chat(&self, _: &[Message], _: f32, _: u32) -> Result<String, String> {
        Err("LLM 未配置".into())
    }
    fn available(&self) -> bool {
        false
    }
    fn supports_vision(&self) -> bool {
        false
    }
}

#[derive(Debug, Clone)]
pub struct OllamaClient {
    pub host: String,
    pub model: String,
    pub timeout: Duration,
}

impl Default for OllamaClient {
    fn default() -> Self {
        Self {
            // Ollama 的标准端口是 11434（9931 是 llama-server 的端口，不是 Ollama 的）。
            // 地址写错会连不上，所有自由问答与看图旁路都会静默退回兜底话术。
            host: "http://127.0.0.1:11434".into(),
            model: "qwen3.5:4b".into(),
            // 本地 4B 模型带思维链，生成一次可达 40~60s，读超时必须留足，否则
            // 读响应被掐断，又退回「答不上来」。
            timeout: Duration::from_secs(120),
        }
    }
}

impl OllamaClient {
    fn authority(&self) -> String {
        self.host
            .trim_start_matches("http://")
            .trim_start_matches("https://")
            .trim_end_matches('/')
            .to_string()
    }
}

impl LlmClient for OllamaClient {
    fn chat(
        &self,
        messages: &[Message],
        temperature: f32,
        num_predict: u32,
    ) -> Result<String, String> {
        let authority = self.authority();
        let addr = authority
            .to_socket_addrs()
            .map_err(|e| format!("解析 Ollama 地址失败: {e}"))?
            .next()
            .ok_or_else(|| "Ollama 地址无解析结果".to_string())?;

        let body = serde_json::json!({
            "model": self.model,
            "messages": messages,
            "stream": false,
            "think": false,
            // 模型常驻显存：否则 Ollama 默认 5 分钟后卸载，下一次提问要重新
            // 冷加载（数秒~十几秒），用户感知为「文字生成很慢」。
            "keep_alive": -1,
            "options": { "temperature": temperature, "num_predict": num_predict }
        })
        .to_string();

        let mut stream = TcpStream::connect_timeout(&addr, self.timeout)
            .map_err(|e| format!("连接 Ollama 失败: {e}"))?;
        let started = std::time::Instant::now();
        stream
            .set_read_timeout(Some(self.timeout))
            .and_then(|_| stream.set_write_timeout(Some(self.timeout)))
            .map_err(|e| format!("设置超时失败: {e}"))?;

        write!(
            stream,
            "POST /api/chat HTTP/1.1\r\nHost: {authority}\r\n\
             Content-Type: application/json\r\nContent-Length: {}\r\n\
             Connection: close\r\n\r\n{body}",
            body.as_bytes().len()
        )
        .map_err(|e| format!("写入请求失败: {e}"))?;
        stream.flush().map_err(|e| format!("刷新请求失败: {e}"))?;

        let mut raw = Vec::new();
        stream
            .read_to_end(&mut raw)
            .map_err(|e| format!("读取响应失败: {e}"))?;
        let text = String::from_utf8_lossy(&raw);
        let split = text
            .find("\r\n\r\n")
            .ok_or_else(|| "Ollama 响应缺少 HTTP 头分隔".to_string())?;
        let payload = &text[split + 4..];
        let value: serde_json::Value =
            serde_json::from_str(payload).map_err(|e| format!("解析响应失败: {e}"))?;
        tracing::debug!(
            host = %self.host,
            model = %self.model,
            messages = messages.len(),
            elapsed_ms = started.elapsed().as_millis() as u64,
            "Ollama 请求完成"
        );
        Ok(value["message"]["content"]
            .as_str()
            .unwrap_or("")
            .trim()
            .to_string())
    }
}

/// llama.cpp server（llama-server）客户端：走 OpenAI 兼容的
/// `POST /v1/chat/completions` 端点。与 Ollama 不同，llama-server
/// 不提供 `/api/chat`，所以需要单独实现。纯文本 GGUF，不支持视觉。
#[derive(Debug, Clone)]
pub struct LlamaServerClient {
    pub host: String,
    pub model: String,
    pub timeout: Duration,
}

impl LlmClient for LlamaServerClient {
    fn chat(
        &self,
        messages: &[Message],
        temperature: f32,
        num_predict: u32,
    ) -> Result<String, String> {
        let authority = self.authority();
        let addr = authority
            .to_socket_addrs()
            .map_err(|e| format!("解析 llama-server 地址失败: {e}"))?
            .next()
            .ok_or_else(|| "llama-server 地址无解析结果".to_string())?;

        let mut body_map = serde_json::Map::new();
        body_map.insert("model".to_string(), serde_json::Value::String(self.model.clone()));
        body_map.insert(
            "messages".to_string(),
            serde_json::to_value(messages).unwrap_or_default(),
        );
        body_map.insert("stream".to_string(), serde_json::Value::Bool(false));
        body_map.insert(
            "temperature".to_string(),
            serde_json::to_value(temperature).unwrap_or_default(),
        );
        if num_predict > 0 {
            body_map.insert(
                "max_tokens".to_string(),
                serde_json::to_value(num_predict).unwrap_or_default(),
            );
        }
        let body = serde_json::Value::Object(body_map).to_string();

        let mut stream = TcpStream::connect_timeout(&addr, self.timeout)
            .map_err(|e| format!("连接 llama-server 失败: {e}"))?;
        let started = std::time::Instant::now();
        stream
            .set_read_timeout(Some(self.timeout))
            .and_then(|_| stream.set_write_timeout(Some(self.timeout)))
            .map_err(|e| format!("设置超时失败: {e}"))?;

        write!(
            stream,
            "POST /v1/chat/completions HTTP/1.1\r\nHost: {authority}\r\n\
             Content-Type: application/json\r\nContent-Length: {}\r\n\
             Connection: close\r\n\r\n{body}",
            body.as_bytes().len()
        )
        .map_err(|e| format!("写入请求失败: {e}"))?;
        stream.flush().map_err(|e| format!("刷新请求失败: {e}"))?;

        let mut raw = Vec::new();
        stream.read_to_end(&mut raw).map_err(|e| format!("读取响应失败: {e}"))?;
        let text = String::from_utf8_lossy(&raw);
        let split = text
            .find("\r\n\r\n")
            .ok_or_else(|| "llama-server 响应缺少 HTTP 头分隔".to_string())?;
        let payload = &text[split + 4..];
        let value: serde_json::Value =
            serde_json::from_str(payload).map_err(|e| format!("解析响应失败: {e}"))?;
        tracing::debug!(
            host = %self.host,
            model = %self.model,
            messages = messages.len(),
            elapsed_ms = started.elapsed().as_millis() as u64,
            "llama-server 请求完成"
        );
        Ok(value["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("")
            .trim()
            .to_string())
    }

    fn supports_vision(&self) -> bool {
        false
    }
}

impl LlamaServerClient {
    fn authority(&self) -> String {
        self.host
            .trim_start_matches("http://")
            .trim_start_matches("https://")
            .trim_end_matches('/')
            .to_string()
    }
}
