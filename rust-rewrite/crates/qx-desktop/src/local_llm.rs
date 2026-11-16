//! 本地 GGUF 大模型（内嵌推理，无需 Ollama / 云端）。
//!
//! 仅在 `cargo build --features local-llm` 时参与编译。Local 版默认加载
//! exe 内嵌的 Qwen2.5-VL-3B（主模型 + mmproj 多模态投影器），文本与看图
//! 都在本进程用 llama.cpp 完成；也可以 `--model <GGUF>` 换外置纯文本模型。
//!
//! 知识库由 qx-core 的 `Guide` 统一写进 system prompt；本模块只负责推理，
//! 不再自己保存/追加知识库，避免「主题切换后旧 KB 残留」和 prompt 重复。
//!
//! 线程安全：模型用 `Box::leak` 提升为 `'static`（进程级单例，仅加载一次；
//! 主题切换不会重建本对象），`LlamaContext` / `MtmdContext` 的求值都不是
//! 线程安全的，统一包进 Mutex。

use std::num::NonZeroU32;
use std::path::Path;
use std::pin::pin;
use std::sync::Mutex;
use std::thread;

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::mtmd::{
    mtmd_default_marker, MtmdBitmap, MtmdContext, MtmdContextParams, MtmdInputText,
};
use llama_cpp_2::sampling::LlamaSampler;
use qx_core::llm::{LlmClient, Message};

/// 知识库 + 图片 token 会占掉不少上下文，8K 比原来的 4K 更稳。
/// llama.cpp 要求单次 decode 的 token 数不超过 n_batch，超过会触发
/// `GGML_ASSERT(n_tokens_all <= n_batch)`，因此 chat_text 按 N_BATCH 分块喂入。
const N_CTX: i32 = 8192;
const N_BATCH: i32 = 4096;

pub struct LocalLlm {
    backend: LlamaBackend,
    // 用 Box::leak 把模型提升为 'static，从而让 LlamaContext<'static> 可存进结构体。
    model: &'static LlamaModel,
    /// 多模态投影器；None = 纯文本模型，不能看图。
    mtmd: Option<Mutex<MtmdContext>>,
    n_ctx: i32,
    n_threads: i32,
}

impl LocalLlm {
    /// 加载本地 GGUF。`mmproj` 给定时启用 mtmd 视觉；加载失败只降级视觉，
    /// 不让整个问答引擎起不来（现场纯文本仍可用）。
    pub fn load(path: &Path, mmproj: Option<&Path>) -> Result<Self, String> {
        if !path.exists() {
            return Err(format!("本地模型文件不存在：{}", path.display()));
        }
        let backend = LlamaBackend::init().map_err(|e| format!("llama 后端初始化失败：{e}"))?;

        // n_gpu_layers = 0 → 纯 CPU 推理（显存留给数字人 / Ollama，避免 8 GB 卡爆掉）。
        let model_params = LlamaModelParams::default();
        let model_params = pin!(model_params);
        let model = LlamaModel::load_from_file(&backend, path, &model_params)
            .map_err(|e| format!("模型加载失败：{e}"))?;
        let model: &'static LlamaModel = Box::leak(Box::new(model));

        let n_threads = thread::available_parallelism()
            .map(|n| n.get().min(8) as i32)
            .unwrap_or(4);
        let mtmd = match mmproj {
            Some(p) => {
                let p_str = crate::embedded_model::ascii_path(p);
                let mut params = MtmdContextParams::default();
                params.use_gpu = false;
                params.n_threads = n_threads;
                match MtmdContext::init_from_file(&p_str, model, &params) {
                    Ok(c) => {
                        tracing::info!("mmproj 就绪，视觉推理在本进程完成");
                        Some(Mutex::new(c))
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, "mmproj 加载失败，看图请求将降级");
                        None
                    }
                }
            }
            None => None,
        };

        Ok(Self {
            backend,
            model,
            mtmd,
            n_ctx: N_CTX,
            n_threads,
        })
    }

    /// 每次问答新建 context：llama-cpp-2 0.1.161 没有公开的「清 KV cache」API，
    /// 而 Qwen2.5-VL 用 M-RoPE，复用上一轮残留位置的 context 会触发
    /// `X < Y` 校验失败（表现为看图请求秒回兜底话术）。新建 context 是最稳的
    /// 公开做法，代价只是每轮重新分配一次 KV（展厅问答频率低，可忽略）。
    fn fresh_context(&self) -> Result<llama_cpp_2::context::LlamaContext<'static>, String> {
        let params = LlamaContextParams::default()
            .with_n_ctx(Some(NonZeroU32::new(N_CTX as u32).unwrap()))
            .with_n_batch(N_BATCH as u32)
            .with_n_threads(self.n_threads);
        self.model
            .new_context(&self.backend, params)
            .map_err(|e| format!("创建上下文失败：{e}"))
    }

    /// 按 Qwen ChatML 拼提示词。知识库已由 qx-core 的 Guide 写进 system 消息，
    /// 这里不再重复注入；带图时在最后一条 user 内容后追加 N 个 media marker。
    fn build_prompt(&self, messages: &[Message], image_count: usize) -> String {
        let marker = mtmd_default_marker();
        let mut s = String::new();
        let last = messages.len().saturating_sub(1);
        for (i, m) in messages.iter().enumerate() {
            s.push_str(&format!("<|im_start|>{}", m.role));
            s.push('\n');
            s.push_str(&m.content);
            if i == last && image_count > 0 {
                for _ in 0..image_count {
                    s.push_str(marker);
                }
            }
            s.push_str("<|im_end|>\n");
        }
        s.push_str("<|im_start|>assistant\n");
        s
    }

    /// 兜底剥离模型仍可能吐出的思维链（含被 num_predict 截断的未闭合块）。
    fn strip_think(s: &str) -> String {
        let mut out = s.to_string();
        loop {
            let Some(start) = out.find("<think>") else { break };
            match out[start..].find("</think>") {
                Some(rel_end) => {
                    let end = start + rel_end + "</think>".len();
                    out.replace_range(start..end, "");
                }
                None => {
                    out.truncate(start);
                    break;
                }
            }
        }
        out.trim().to_string()
    }

    fn sampler(&self, temperature: f32) -> LlamaSampler {
        let temp = if temperature <= 0.0 { 0.0 } else { temperature };
        if temp == 0.0 {
            LlamaSampler::chain_simple([LlamaSampler::top_k(40), LlamaSampler::greedy()])
        } else {
            LlamaSampler::chain_simple([
                LlamaSampler::temp(temp),
                LlamaSampler::top_k(40),
                LlamaSampler::top_p(0.9, 1),
                LlamaSampler::dist(1234),
            ])
        }
    }

    /// 纯文本路径：tokenize → 逐 token decode → 采样，与原实现一致。
    fn chat_text(
        &self,
        messages: &[Message],
        temperature: f32,
        num_predict: u32,
    ) -> Result<String, String> {
        let prompt = self.build_prompt(messages, 0);
        let vocab = self.model.vocab();
        let tokens = vocab.tokenize(prompt.as_bytes(), true, true);
        if tokens.is_empty() {
            return Err("提示词分词为空".into());
        }
        let prompt_len = tokens.len() as i32;
        if prompt_len >= self.n_ctx {
            return Err("提示词超出上下文窗口".into());
        }

        let mut ctx = self.fresh_context()?;
        // 按 n_batch 分块喂入：单次 decode 超过 n_batch 会 GGML_ASSERT。
        // 只在最后一个 token 上要 logits，其余块纯 prefill。
        let mut batch = LlamaBatch::new(N_BATCH as usize, 1);
        for (chunk_idx, chunk) in tokens.chunks(N_BATCH as usize).enumerate() {
            batch.clear();
            let base = chunk_idx as i32 * N_BATCH;
            for (i, t) in chunk.iter().enumerate() {
                let pos = base + i as i32;
                let is_last = pos == prompt_len - 1;
                batch
                    .add(*t, pos, &[0], is_last)
                    .map_err(|e| format!("batch 添加失败：{e}"))?;
            }
            ctx.decode(&mut batch)
                .map_err(|e| format!("首轮解码失败：{e}"))?;
        }

        let mut sampler = self.sampler(temperature);
        let mut out_tokens = Vec::new();
        let mut n_cur = prompt_len;
        let max_out = num_predict as i32;
        loop {
            let token;
            {
                token = sampler.sample(&ctx, batch.n_tokens() - 1);
                sampler.accept(token);
                if vocab.is_eog(token) {
                    break;
                }
                batch.clear();
                batch
                    .add(token, n_cur, &[0], true)
                    .map_err(|e| format!("batch 续写失败：{e}"))?;
                n_cur += 1;
                ctx.decode(&mut batch)
                    .map_err(|e| format!("解码失败：{e}"))?;
            }
            out_tokens.push(token);
            if out_tokens.len() as i32 >= max_out || n_cur + 1 > self.n_ctx {
                break;
            }
        }
        let bytes = vocab.detokenize(&out_tokens, true, true);
        Ok(Self::strip_think(&String::from_utf8_lossy(&bytes)))
    }

    /// 视觉路径：base64 → stb_image 解码 → mtmd tokenize → eval_chunks → 采样。
    /// `logits_last=true` 时最后一次 decode 只输出一个 token 的 logits，
    /// 因此采样索引用 0（见 mtmd-helper.cpp 的 set_output_logits 逻辑）。
    fn chat_vision(
        &self,
        messages: &[Message],
        images: &[&String],
        temperature: f32,
        num_predict: u32,
    ) -> Result<String, String> {
        let mtmd_mutex = self.mtmd.as_ref().expect("调用方已检查");
        let mut mtmd = mtmd_mutex.lock().unwrap();

        let mut bitmaps = Vec::with_capacity(images.len());
        for b64 in images {
            let bytes = b64_decode(b64).ok_or("图片 base64 无法解码")?;
            let bm = MtmdBitmap::from_buffer(&*mtmd, &bytes, false)
                .map_err(|e| format!("图片解码失败：{e:?}"))?;
            bitmaps.push(bm);
        }
        let refs: Vec<&MtmdBitmap> = bitmaps.iter().collect();
        let prompt = self.build_prompt(messages, bitmaps.len());
        let chunks = mtmd
            .tokenize(
                MtmdInputText {
                    text: prompt,
                    add_special: true,
                    parse_special: true,
                },
                &refs,
            )
            .map_err(|e| format!("多模态分词失败：{e:?}"))?;

        let vocab = self.model.vocab();
        let mut sampler = self.sampler(temperature);
        let mut out_tokens = Vec::new();
        let mut ctx = self.fresh_context()?;
        {
            let mut pos = chunks
                .eval_chunks(&mut *mtmd, &mut ctx, 0, 0, N_BATCH, true)
                .map_err(|e| format!("多模态解码失败：{e:?}"))?;
            let mut batch = LlamaBatch::new(self.n_ctx as usize, 1);
            loop {
                // mtmd 的 text chunk 按 n_batch 切分，最后一个有 logits 的 token 的
                // batch 内索引不固定，不能像纯文本路径那样用 batch.n_tokens()-1。
                // 用无索引的 get_logits() 构造候选（紧凑取「最后一个有 logits 的 token」）。
                let mut data = ctx.token_data_array();
                sampler.apply(&mut data);
                let token = data.selected_token().ok_or("视觉采样失败：无候选")?;
                sampler.accept(token);
                if vocab.is_eog(token) {
                    break;
                }
                batch.clear();
                batch
                    .add(token, pos, &[0], true)
                    .map_err(|e| format!("batch 续写失败：{e}"))?;
                pos += 1;
                ctx.decode(&mut batch)
                    .map_err(|e| format!("解码失败：{e}"))?;
                out_tokens.push(token);
                if out_tokens.len() as i32 >= num_predict as i32 || pos >= self.n_ctx {
                    break;
                }
            }
        }
        let bytes = vocab.detokenize(&out_tokens, true, true);
        Ok(Self::strip_think(&String::from_utf8_lossy(&bytes)))
    }
}

/// 标准 base64 解码（容忍 `data:image/...;base64,` 前缀与空白）。
/// 不引第三方 crate：mtmd 自带 stb_image，拿到文件字节就能直接解码。
fn b64_decode(s: &str) -> Option<Vec<u8>> {
    let s = s.trim();
    let body = s.split_once(',').map(|x| x.1).unwrap_or(s);
    let mut out = Vec::with_capacity(body.len() * 3 / 4);
    let mut acc = 0u32;
    let mut bits = 0u32;
    for c in body.bytes() {
        let v = match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            b'=' => break,
            b'\r' | b'\n' | b' ' => continue,
            _ => return None,
        };
        acc = (acc << 6) | v as u32;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
        }
    }
    Some(out)
}

impl LlmClient for LocalLlm {
    fn chat(
        &self,
        messages: &[Message],
        temperature: f32,
        num_predict: u32,
    ) -> Result<String, String> {
        let images: Vec<&String> = messages.iter().flat_map(|m| m.images.iter()).collect();
        if images.is_empty() {
            self.chat_text(messages, temperature, num_predict)
        } else if self.mtmd.is_some() {
            self.chat_vision(messages, &images, temperature, num_predict)
        } else {
            Err("本地 GGUF 没有 mmproj 视觉投影器，不支持图片输入".into())
        }
    }

    fn supports_vision(&self) -> bool {
        self.mtmd.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::b64_decode;

    #[test]
    fn b64_roundtrip_known_bytes() {
        assert_eq!(b64_decode("SGVsbG8=").unwrap(), b"Hello");
        assert_eq!(b64_decode("AP8=").unwrap(), vec![0, 255]);
    }

    #[test]
    fn b64_tolerates_data_url() {
        assert_eq!(
            b64_decode("data:image/png;base64,SGVsbG8=").unwrap(),
            b"Hello"
        );
    }

    #[test]
    fn b64_rejects_garbage() {
        assert!(b64_decode("@@@").is_none());
    }
}
