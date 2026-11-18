//! Local 版的视觉旁路。
//!
//! Local 版用纯文本 GGUF 内嵌推理，没有视觉投影器（mmproj），所以「拖图提问」
//! 在本机模型上必然答不了。但用户机器上若装有带视觉的 Ollama 模型（如
//! `qwen2.5vl:3b`），完全可以「文字仍走本地 GGUF，图片借 Ollama 看一眼」——
//! 这样既不动内嵌模型的轻量路线，也不让 Local 版在演示时丢掉看图能力。
//!
//! 关键设计：
//! 1. 视觉旁路是**可选的**：探测不到 Ollama 或没有 VL 模型时，行为与原来完全一致
//!    （`supports_vision()` 为 false，界面照旧提示「当前模型不支持看图」）；
//! 2. 只在消息里真的带图时才转过去，纯文字永远走主客户端，不引入额外延迟；
//! 3. 旁路调用失败就回落主客户端，让上层拿到原本的降级话术，不抛错给用户。

use qx_core::llm::{LlmClient, Message, OllamaClient};
use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::sync::Arc;
use std::time::Duration;

/// 从 `/api/tags` 的响应里挑一个带视觉的模型名。
///
/// 优先按 `capabilities` 里是否含 `vision` 判定（新版 Ollama 的 `/api/tags`
/// 会直接返回该字段，最可靠）；老版本 Ollama 没有 `capabilities` 时，退回按
/// 名字关键字匹配（`vl` / `vision` / `llava` 等）。挑不到就返回 None（宁可不
/// 启用，也不误用一个纯文本模型）。
pub fn pick_vision_model(tags_json: &str) -> Option<String> {
    let v: serde_json::Value = serde_json::from_str(tags_json).ok()?;
    let models = v.get("models")?.as_array()?;

    // 1) 带 capabilities 字段：找第一个声明 vision 的模型。
    for m in models {
        let caps = m.get("capabilities").and_then(|c| c.as_array());
        let has_vision = caps
            .map(|caps| caps.iter().any(|c| c.as_str() == Some("vision")))
            .unwrap_or(false);
        if has_vision {
            if let Some(n) = m.get("name").and_then(|x| x.as_str()) {
                return Some(n.to_string());
            }
        }
    }

    // 2) 回退：老 Ollama 没有 capabilities，按名字关键字匹配。
    let names: Vec<String> = models
        .iter()
        .filter_map(|m| m.get("name").and_then(|x| x.as_str()).map(|s| s.to_string()))
        .collect();
    for key in ["vl", "vision", "llava", "minicpm-v", "moondream"] {
        if let Some(n) = names.iter().find(|n| n.to_lowercase().contains(key)) {
            return Some(n.clone());
        }
    }
    None
}

/// 从原始响应里取出 body，兼容 `Transfer-Encoding: chunked`。
///
/// Go 的 http server（Ollama 就是）在不知道长度时用 chunked，直接按
/// `\r\n\r\n` 切开拿到的是一堆 `1a2b\r\n{...}\r\n0\r\n\r\n`，JSON 解析必然失败。
fn http_body(raw: &str) -> Option<String> {
    let (head, body) = raw.split_once("\r\n\r\n")?;
    let chunked = head
        .to_lowercase()
        .lines()
        .any(|l| l.starts_with("transfer-encoding:") && l.contains("chunked"));
    if !chunked {
        return Some(body.to_string());
    }
    let mut out = String::new();
    let mut rest = body;
    loop {
        let (size_line, after) = rest.split_once("\r\n")?;
        let size = usize::from_str_radix(size_line.trim().split(';').next()?, 16).ok()?;
        if size == 0 {
            break;
        }
        if after.len() < size {
            out.push_str(after);
            break;
        }
        out.push_str(&after[..size]);
        rest = after[size..].strip_prefix("\r\n").unwrap_or(&after[size..]);
    }
    Some(out)
}

/// 探测本机 Ollama 是否可用，并返回一个指向 VL 模型的客户端。
pub fn probe_vision(host: &str) -> Option<OllamaClient> {
    let authority = host
        .trim_start_matches("http://")
        .trim_start_matches("https://")
        .trim_end_matches('/')
        .to_string();
    let addr = authority.to_socket_addrs().ok()?.next()?;

    let mut s = TcpStream::connect_timeout(&addr, Duration::from_millis(600)).ok()?;
    s.set_read_timeout(Some(Duration::from_millis(900))).ok()?;
    s.set_write_timeout(Some(Duration::from_millis(600))).ok()?;
    write!(
        s,
        "GET /api/tags HTTP/1.1\r\nHost: {authority}\r\nConnection: close\r\n\r\n"
    )
    .ok()?;
    s.flush().ok()?;

    let mut raw = Vec::new();
    // 不能直接用 read_to_end：Ollama（Go 的 http server）不一定会按
    // `Connection: close` 立刻关连接，read_to_end 会一直等到读超时并返回 Err，
    // 把已经读到的字节整段丢掉 —— 探测就会「明明服务在跑却探不到」。
    let mut buf = [0u8; 4096];
    loop {
        match s.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => {
                raw.extend_from_slice(&buf[..n]);
                if raw.len() > 512 * 1024 {
                    break;
                }
            }
            Err(_) => break, // 超时也保留已读内容
        }
    }
    let text = String::from_utf8_lossy(&raw);
    let body = http_body(&text)?;
    let model = pick_vision_model(&body)?;
    Some(OllamaClient {
        host: format!("http://{authority}"),
        model,
        timeout: Duration::from_secs(45),
    })
}

/// 主客户端 + 可选视觉旁路。
pub struct VisionRoute {
    primary: Arc<dyn LlmClient>,
    fallback: Option<OllamaClient>,
}

impl VisionRoute {
    pub fn new(primary: Arc<dyn LlmClient>, fallback: Option<OllamaClient>) -> Self {
        Self { primary, fallback }
    }
    /// 供日志/自检读取旁路所用的模型名。
    pub fn fallback_model(&self) -> Option<&str> {
        self.fallback.as_ref().map(|f| f.model.as_str())
    }
}

impl LlmClient for VisionRoute {
    fn chat(&self, messages: &[Message], temperature: f32, num_predict: u32) -> Result<String, String> {
        let has_image = messages.iter().any(|m| !m.images.is_empty());
        if has_image {
            if let Some(fb) = &self.fallback {
                match fb.chat(messages, temperature, num_predict) {
                    Ok(s) if !s.trim().is_empty() => return Ok(s),
                    _ => {} // 旁路失败：静默回落，交给上层走降级话术
                }
            }
        }
        self.primary.chat(messages, temperature, num_predict)
    }

    fn available(&self) -> bool {
        self.primary.available()
    }

    fn supports_vision(&self) -> bool {
        self.fallback.is_some() || self.primary.supports_vision()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    struct Fake {
        name: &'static str,
        vision: bool,
        seen: Arc<Mutex<Vec<bool>>>, // 记录每次调用是否带图
        fail: bool,
    }

    impl LlmClient for Fake {
        fn chat(&self, messages: &[Message], _: f32, _: u32) -> Result<String, String> {
            self.seen
                .lock()
                .unwrap()
                .push(messages.iter().any(|m| !m.images.is_empty()));
            if self.fail {
                Err("boom".into())
            } else {
                Ok(self.name.to_string())
            }
        }
        fn available(&self) -> bool {
            true
        }
        fn supports_vision(&self) -> bool {
            self.vision
        }
    }

    #[test]
    fn picks_vl_model() {
        let tags = r#"{"models":[{"name":"qwen3.5:4b"},{"name":"qwen2.5vl:3b"}]}"#;
        assert_eq!(pick_vision_model(tags).as_deref(), Some("qwen2.5vl:3b"));
        let none = r#"{"models":[{"name":"qwen3.5:4b"}]}"#;
        assert_eq!(pick_vision_model(none), None);
    }

    #[test]
    fn picks_model_by_capabilities() {
        // 新版 Ollama 的 /api/tags 带 capabilities；模型名叫 qwen3.5:4b 但声明了
        // vision，必须能按能力挑出来——否则「图片也无法识别」会一直复发。
        let tags = r#"{"models":[
            {"name":"qwen-local:latest","capabilities":["tools","thinking","completion","vision"]},
            {"name":"qwen3.5:4b","capabilities":["tools","thinking","completion","vision"]}
        ]}"#;
        assert_eq!(pick_vision_model(tags).as_deref(), Some("qwen-local:latest"));
        // capabilities 里没有 vision 的纯文本模型不能被误挑
        let text_only = r#"{"models":[{"name":"qwen3.5:4b","capabilities":["tools","completion"]}]}"#;
        assert_eq!(pick_vision_model(text_only), None);
    }

    #[test]
    fn routes_images_to_fallback() {
        let p_seen = Arc::new(Mutex::new(Vec::new()));
        let f_seen: Arc<Mutex<Vec<bool>>> = Arc::new(Mutex::new(Vec::new()));
        let primary = Arc::new(Fake { name: "local", vision: false, seen: p_seen.clone(), fail: false });
        let fb = OllamaClient { host: "http://x".into(), model: "qwen2.5vl:3b".into(), timeout: Duration::from_secs(1) };
        // 这里只验证路由判定，不真的发请求：用一个能返回 Err 的假 Ollama 地址，
        // 失败后会回落到 primary，正好验证第 3 条设计。
        let route = VisionRoute::new(primary, Some(fb));
        assert!(route.supports_vision());

        let text_only = vec![Message::user("你好")];
        assert_eq!(route.chat(&text_only, 0.2, 32).unwrap(), "local");
        assert_eq!(*p_seen.lock().unwrap(), vec![false]);

        let with_img = vec![Message::user("看图").with_images(vec!["AAA".into()])];
        // 地址不可达 → 旁路 Err → 回落 primary，且 primary 收到的是同一条带图消息
        assert_eq!(route.chat(&with_img, 0.2, 32).unwrap(), "local");
        assert_eq!(*p_seen.lock().unwrap(), vec![false, true]);
        assert!(f_seen.lock().unwrap().is_empty());
    }

    #[test]
    fn no_fallback_keeps_old_behaviour() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let primary = Arc::new(Fake { name: "local", vision: false, seen, fail: false });
        let route = VisionRoute::new(primary, None);
        assert!(!route.supports_vision());
    }

    #[test]
    fn decodes_chunked_body() {
        // 分块长度必须是实际字节数，手写常量最容易错——这里按 len() 现算。
        let a = "{\"models\":[{\"name\"";
        let b = ":\"qwen2.5vl:3b\"}]}";
        let raw = format!(
            "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n{:x}\r\n{}\r\n{:x}\r\n{}\r\n0\r\n\r\n",
            a.len(),
            a,
            b.len(),
            b
        );
        let body = http_body(&raw).expect("body");
        assert_eq!(body, r#"{"models":[{"name":"qwen2.5vl:3b"}]}"#);
        assert_eq!(pick_vision_model(&body).as_deref(), Some("qwen2.5vl:3b"));
    }

    /// 用本地假服务器验证整套探测：**不依赖本机 Ollama 是否在跑**。
    /// 之所以要假服务器：Ollama 是 Go 写的，`/api/tags` 常以 chunked 返回，
    /// 且在本机会话里进程生命周期不受控，真机测试既不稳定也复现不了 chunked。
    ///
    /// 并行跑全套测试时，假服务器线程可能来不及在探测端的 900ms 读超时内
    /// 响应（调度被重负载拖慢），因此探测带重试、服务器循环服务多次连接。
    #[test]
    fn probe_against_fake_ollama() {
        use std::io::{Read as _, Write as _};
        use std::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let handle = std::thread::spawn(move || {
            // 循环服务，配合客户端重试；空闲约 1 秒（50 × 20ms 无连接）即退出，
            // 保证 join() 能返回——固定服务 N 次会在客户端提前成功时永远等不满。
            listener.set_nonblocking(true).ok();
            let mut idle = 0;
            loop {
                match listener.accept() {
                    Ok((mut c, _)) => {
                        idle = 0;
                        let mut buf = [0u8; 1024];
                        let _ = c.read(&mut buf);
                        let body = r#"{"models":[{"name":"qwen3.5:4b"},{"name":"qwen2.5vl:3b"}]}"#;
                        let resp = format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\
                             Transfer-Encoding: chunked\r\n\r\n{:x}\r\n{}\r\n0\r\n\r\n",
                            body.len(),
                            body
                        );
                        let _ = c.write_all(resp.as_bytes());
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        idle += 1;
                        if idle > 50 {
                            break;
                        }
                        std::thread::sleep(std::time::Duration::from_millis(20));
                    }
                    Err(_) => break,
                }
            }
        });

        let host = format!("http://127.0.0.1:{port}");
        let mut client = None;
        for _ in 0..8 {
            if let Some(c) = probe_vision(&host) {
                client = Some(c);
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(60));
        }
        let client = client.expect("应探测到 VL 模型");
        assert_eq!(client.model, "qwen2.5vl:3b");
        assert_eq!(client.host, host);
        let _ = handle.join();
    }
}
