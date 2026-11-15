//! 知识库检索与话术路由，行为对齐 `server/nlu.py` 的六级路由。
//!
//! 一处有意的**优化**：原版用 jieba 分词 + pypinyin 同音纠错。重写版改为
//! 「主题包领域词典最长匹配 + 汉字双字组」，短查询不再被通用分词器切错，
//! 也去掉了 jieba / pypinyin / 词典文件三项运行时依赖。别名、FAQ、实时信息、
//! 欢迎与路线等主要分支本来就走子串匹配，行为保持一致。
//!
//! 尚未移植（阶段二）：拼音同音纠错 `correct()`、会话级 LLM 上下文、
//! 带图视觉问答。这些不影响知识库主链路。

use crate::llm::{LlmClient, Message, NoLlm};
use crate::theme::{Exhibit, Theme};
use regex::Regex;
use std::collections::HashSet;
use std::sync::Arc;

const DEFAULT_REALTIME: &str = "这类实时信息我查不到——我是本机离线跑的讲解员，没有联网看天气、看新闻。侨乡的事你尽管问，碉楼、骑楼、陈皮、侨批我都熟。";

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct Answer {
    pub reply: String,
    pub action: String,
    pub emotion: String,
    pub gesture: String,
    pub exhibit: Option<String>,
    pub exhibit_name: Option<String>,
}

pub struct Guide {
    pub theme: Theme,
    llm: Arc<dyn LlmClient>,
    dict: HashSet<String>,
    stop: HashSet<String>,
    ex_tokens: Vec<HashSet<String>>,
    faq_tokens: Vec<Vec<HashSet<String>>>,
    punct: Regex,
    realtime: Regex,
    focus_rules: Vec<(&'static str, Regex)>,
    what: Regex,
    md: Regex,
    redirect_hint: Regex,
    redirect_topic: Regex,
}

fn rx(pattern: &str) -> Regex {
    Regex::new(pattern).expect("内置正则应为常量且合法")
}

fn char_len(s: &str) -> usize {
    s.chars().count()
}

fn overlap(a: &HashSet<String>, b: &HashSet<String>) -> f64 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    a.intersection(b).count() as f64 / a.len() as f64
}

impl Guide {
    pub fn new(theme: Theme) -> Self {
        Self::with_llm(theme, Arc::new(NoLlm))
    }

    pub fn with_llm(theme: Theme, llm: Arc<dyn LlmClient>) -> Self {
        let punct = rx(r"[，。！？、；：,.!?;:\s“”‘’()（）《》【】]");
        let realtime = rx("天气|气温|下雨|降雨|台风|温度几度|空气质量|雾霾|新闻|热搜|头条|最近发生|刚刚发生|股价|股票|汇率|比特币|基金|油价|房价|现在几点|今天几号|今天星期几|几月几号|现在时间|比分|赛况|航班|高铁票|路况|堵不堵");
        let focus_rules = vec![
            ("why", rx("为什么|为啥|原因|怎么会")),
            ("howmany", rx("多少|多高|多长|多大|多远|多久|几年|几座|几个|几次|几层|多重|几岁")),
            ("when", rx("哪年|什么时候|何时|几几年|哪个朝代|哪一代")),
            ("who", rx("是谁|谁住|谁盖|谁建|哪个人|什么人|有哪些人")),
            ("where", rx("在哪|哪里|哪儿|什么地方")),
            ("how", rx("怎么|如何|怎样")),
        ];
        let stop: HashSet<String> = "的 了 是 在 我 你 他 它 这 那 有 和 与 就 都 也 吗 呢 啊 吧 请问 一下 什么 怎么 为什么 能 可以 想 要 个 件 些 谁"
            .split_whitespace()
            .map(str::to_string)
            .collect();

        let mut dict = HashSet::new();
        for e in &theme.exhibits {
            for w in e.aliases.iter().chain(e.keywords.iter()) {
                if char_len(w) >= 2 {
                    dict.insert(w.clone());
                }
            }
        }
        for c in &theme.chitchat {
            for k in &c.k {
                if char_len(k) >= 2 {
                    dict.insert(k.clone());
                }
            }
        }

        let mut g = Self {
            theme,
            llm,
            dict,
            stop,
            ex_tokens: Vec::new(),
            faq_tokens: Vec::new(),
            punct,
            realtime,
            focus_rules,
            what: rx("是什么|什么是|啥|什么意思"),
            md: rx(r"[*#`>~|]+"),
            redirect_hint: rx("感兴趣|想不想|要不要|可以问我|还可以问|换个话题|关系不大|有点远|带你去看"),
            redirect_topic: rx("侨乡|江门|碉楼|骑楼|展品|展厅|小鸟天堂|陈皮|侨批"),
        };

        // 预计算检索词，等价于 Python Guide.load() 里构建 _tokens 的做法
        for e in &g.theme.exhibits {
            let mut set = HashSet::new();
            for w in e.keywords.iter().chain(e.aliases.iter()) {
                set.extend(g.tokenize(w));
            }
            g.ex_tokens.push(set);
            let mut faqs = Vec::new();
            for f in &e.faq {
                faqs.push(g.tokenize(&f.q));
            }
            g.faq_tokens.push(faqs);
        }
        g
    }

    fn tokenize(&self, text: &str) -> HashSet<String> {
        let clean = self.punct.replace_all(text, "").to_string();
        let chars: Vec<char> = clean.chars().collect();
        let mut out = HashSet::new();
        let mut i = 0;
        while i < chars.len() {
            let mut matched = false;
            for len in (2..=5).rev() {
                if i + len <= chars.len() {
                    let cand: String = chars[i..i + len].iter().collect();
                    if self.dict.contains(&cand) {
                        if !self.stop.contains(&cand) {
                            out.insert(cand);
                        }
                        i += len;
                        matched = true;
                        break;
                    }
                }
            }
            if !matched {
                if i + 2 <= chars.len() {
                    let bg: String = chars[i..i + 2].iter().collect();
                    if !self.stop.contains(&bg) {
                        out.insert(bg);
                    }
                }
                i += 1;
            }
        }
        out
    }

    fn focus(&self, text: &str) -> HashSet<&'static str> {
        let mut out = HashSet::new();
        for (name, re) in &self.focus_rules {
            if re.is_match(text) {
                out.insert(*name);
            }
        }
        if out.is_empty() && self.what.is_match(text) {
            out.insert("what");
        }
        out
    }

    /// 对应 Python `_best_faq`：先精确命中原问句，再按疑问焦点过滤后打分。
    fn best_faq(&self, ex_idx: usize, text: &str) -> Option<String> {
        let ex = &self.theme.exhibits[ex_idx];
        let raw_q = self.punct.replace_all(text, "").to_string();
        for f in &ex.faq {
            if self.punct.replace_all(&f.q, "").to_string() == raw_q {
                return Some(f.a.clone());
            }
        }
        let mut tokens = self.tokenize(text);
        for t in &self.ex_tokens[ex_idx] {
            tokens.remove(t);
        }
        let want = self.focus(text);
        let mut best: Option<(f64, String)> = None;
        for (j, f) in ex.faq.iter().enumerate() {
            let have = self.focus(&f.q);
            if !want.is_empty() && !have.is_empty()
                && want.intersection(&have).next().is_none()
            {
                continue;
            }
            let s = overlap(&tokens, &self.faq_tokens[ex_idx][j]);
            if best.as_ref().is_none_or(|(bs, _)| s > *bs) {
                best = Some((s, f.a.clone()));
            }
        }
        best.filter(|(s, _)| *s >= 0.5).map(|(_, a)| a)
    }

    pub fn answer(
        &self,
        text: &str,
        exhibit_id: Option<&str>,
        deep: bool,
        session: Option<&str>,
    ) -> Answer {
        self.answer_with_images(text, exhibit_id, deep, session, &[])
    }

    /// 带图问答入口：前端把压缩后的图片 base64 放进 `images`，这里优先交给视觉模型。
    /// 迁移到 Rust 版时整条 images 链路曾被漏掉，导致拖图后只走文字兜底（「答不上来」）。
    pub fn answer_with_images(
        &self,
        text: &str,
        exhibit_id: Option<&str>,
        deep: bool,
        session: Option<&str>,
        images: &[String],
    ) -> Answer {
        if !images.is_empty() {
            if let Some(reply) = self.vision_answer(text, images, exhibit_id) {
                let ex_idx = exhibit_id
                    .and_then(|id| self.theme.exhibits.iter().position(|e| e.id == id));
                let action = if ex_idx.is_some() { "explain" } else { "chat" };
                return self.pack(reply, action, ex_idx, None);
            }
            if text.trim().is_empty() {
                // 只发了图、而后端没能给出答案：说清原因，别丢一句不相干的兜底话术。
                // 注意 available() 对 Ollama 客户端是「假定可用」，真实调用失败时
                // vision_available() 仍可能为 true，故按是否有视觉能力分两种措辞。
                let msg = if !self.vision_available() {
                    "这张图我暂时看不了：当前模型不支持看图，或本地大模型还没连上。您可以直接用文字问我，我来介绍展厅里的展品。"
                } else {
                    "这张图我暂时看不了，可能是本地大模型正忙或还没连上。请再试一次，也可以直接用文字问我。"
                };
                return self.pack(msg.to_string(), "fallback", None, Some("neutral"));
            }
            // 随图还写了文字：别把文字问题一起丢掉，继续走普通文本问答（对齐旧版 DEF-20）
        }
        self.answer_text(text, exhibit_id, deep, session)
    }

    fn answer_text(
        &self,
        text: &str,
        exhibit_id: Option<&str>,
        deep: bool,
        _session: Option<&str>,
    ) -> Answer {
        let text = text.trim();
        let raw = self.punct.replace_all(text, "").to_string();

        // 0. 空输入 / 实时信息
        if raw.is_empty() {
            return self.pack(self.theme.welcome.clone(), "welcome", None, Some("happy"));
        }
        if self.realtime.is_match(&raw) {
            let msg = if self.theme.realtime.is_empty() {
                DEFAULT_REALTIME.to_string()
            } else {
                self.theme.realtime.clone()
            };
            return self.pack(msg, "fallback", None, Some("neutral"));
        }

        // 1. 闲聊：只认「整句就是在聊这个」，避免吞掉长句
        for c in &self.theme.chitchat {
            for k in &c.k {
                if raw.contains(k) && char_len(&raw) <= char_len(k) + 8 {
                    return self.pack(c.a.clone(), "chat", None, None);
                }
            }
        }

        // 2. 展品实体命中（优先最长别名）
        let mut hit: Option<(usize, usize)> = None;
        for (i, ex) in self.theme.exhibits.iter().enumerate() {
            for a in &ex.aliases {
                let n = char_len(a);
                if raw.contains(a) && hit.is_none_or(|(_, best)| n > best) {
                    hit = Some((i, n));
                }
            }
        }
        if let Some((idx, _)) = hit {
            let ex = self.theme.exhibits[idx].clone();
            let intro = ["介绍", "讲讲", "说说", "讲解", "听听"]
                .iter()
                .any(|k| raw.contains(k));
            if intro {
                let want_deep = deep
                    || ["详细", "深入", "展开"].iter().any(|k| raw.contains(k));
                let body = if want_deep { ex.deep.clone() } else { ex.script.clone() };
                return self.pack(body, "explain", Some(idx), None);
            }
            if let Some(a) = self.best_faq(idx, text) {
                return self.pack(a, "explain", Some(idx), Some("neutral"));
            }
            if let Some(reply) = self.llm_answer(text, Some(&ex), 0.35, false) {
                return self.pack(reply, "explain", Some(idx), None);
            }
            let want_deep = deep
                || ["详细", "深入", "展开", "具体"]
                    .iter()
                    .any(|k| raw.contains(k));
            let body = if want_deep { ex.deep.clone() } else { ex.script.clone() };
            return self.pack(body, "explain", Some(idx), None);
        }

        // 3. 当前聚焦展品的追问（「它」「这个」指代）
        if let Some(id) = exhibit_id {
            if let Some(idx) = self.theme.exhibits.iter().position(|e| e.id == id) {
                if let Some(a) = self.best_faq(idx, text) {
                    return self.pack(a, "explain", Some(idx), None);
                }
                if ["它", "这个", "这件", "还有", "继续", "然后"]
                    .iter()
                    .any(|k| raw.contains(k))
                {
                    let body = self.theme.exhibits[idx].deep.clone();
                    return self.pack(body, "explain", Some(idx), None);
                }
            }
        }

        // 4. 无实体：按关键词打分
        let tokens = self.tokenize(text);
        let mut best: Option<(usize, f64)> = None;
        for (i, ex_tok) in self.ex_tokens.iter().enumerate() {
            let s = overlap(&tokens, ex_tok);
            if best.is_none_or(|(_, bs)| s > bs) {
                best = Some((i, s));
            }
        }
        if let Some((idx, score)) = best {
            if score >= 0.34 {
                let reply = self
                    .best_faq(idx, text)
                    .unwrap_or_else(|| self.theme.exhibits[idx].script.clone());
                return self.pack(reply, "explain", Some(idx), None);
            }
        }

        // 5. 意图
        if ["路线", "顺序", "怎么逛", "参观", "推荐", "先看"]
            .iter()
            .any(|k| raw.contains(k))
        {
            return self.pack(self.theme.route.clone(), "route", None, None);
        }
        if ["你好", "欢迎", "开始", "开场"]
            .iter()
            .any(|k| raw.contains(k))
        {
            return self.pack(self.theme.welcome.clone(), "welcome", None, None);
        }

        // 6. 知识库之外交给本地大模型，失败回落到兜底话术
        if let Some(reply) = self.llm_answer(text, None, 0.65, true) {
            return self.pack(reply, "chat", None, None);
        }
        self.pack(self.theme.fallback.clone(), "fallback", None, None)
    }

    /// 把整个主题包知识库（展品讲解词 + 深度资料 + 问答对 + 路线）原样喂给模型，
    /// 让任何一轮问答都基于已知资料、不臆测。KB 体量很小（6 个展项），直接塞进
    /// system prompt 即可，无需向量库 / RAG 检索（朋友建议的「把数据库以提示词的方式
    /// 喂给 AI」）。注意：只放事实资料，不放 quiz 答案与兜底话术，避免模型直接背答案。
    fn kb_context(&self) -> String {
        let mut s = String::from(
            "## 已知资料（只能依据以下资料作答，资料里没有就如实说不知道，绝不编造）\n",
        );
        for e in &self.theme.exhibits {
            s.push_str(&format!("【{}】\n{}\n{}\n", e.name, e.script, e.deep));
            for f in &e.faq {
                s.push_str(&format!("问：{}　答：{}\n", f.q, f.a));
            }
        }
        if !self.theme.route.is_empty() {
            s.push_str(&format!("参观路线建议：{}\n", self.theme.route));
        }
        s
    }

    fn system_prompt(&self) -> String {
        format!(
            // 自我介绍口径：只报「AI 智能讲解员{名}」，前面不带「{馆名}展厅的」之类前缀
            //（用户明确要求）。同时限定话题边界：只答展馆相关内容，避免像「买围巾」
            // 这类离题编造（比赛现场事故点）。
            "你是 AI 智能讲解员{name}，在{place}为观众讲解。自我介绍时只说「我是 AI 智能讲解员{name}」，\
             不要在前面加「{place}展厅的」等前缀。\
             只谈与{place}的展品、历史、人物、风物和参观相关的内容；观众问别的事\
             （购物、点餐、私人情感、代写文章等），就礼貌说明自己只负责展馆讲解，\
             并把话题引回展品，绝不编造与展馆无关的故事。\
             只用已知资料回答，资料里没有就如实说明，不臆测、不反问。\
             回答不超过 90 字，口语化、不分点、不加表情，像面对面给游客讲解。\n\n{kb}",
            name = self.theme.assistant.name,
            place = self.theme.name,
            kb = self.kb_context(),
        )
    }

    /// 模型是否可用（决定是否走 LLM、以及 /api/stats 的 available 字段）。
    pub fn llm_available(&self) -> bool {
        self.llm.available()
    }

    /// 当前后端能否看图。Ollama 的 qwen3.5:4b 带视觉投影器；纯文本 GGUF 不能。
    pub fn vision_available(&self) -> bool {
        self.llm.available() && self.llm.supports_vision()
    }

    /// 看图回答：视觉模型直接看图，知识库只作为识别参考（对齐迁移前 Python 版语义）。
    /// system 口径与纯文本问答不同——允许如实描述与侨乡无关的画面，不硬拉回展厅。
    fn vision_answer(
        &self,
        text: &str,
        images: &[String],
        exhibit_id: Option<&str>,
    ) -> Option<String> {
        if !self.vision_available() {
            return None;
        }
        let mut system = format!(
            "你是{}的 AI 讲解员{}。先看清图片，再用讲解员的口吻回答。\
             图片若是展厅展品或侨乡风物，就结合已知资料介绍；若与侨乡无关，如实描述、友好回应，不必硬拉回展厅。\
             只讲看得见的内容，拿不准就说拿不准。回答不超过 90 字，不分点、不加表情、不反问。",
            self.theme.name, self.theme.assistant.name,
        );
        let names: Vec<String> = self
            .theme
            .exhibits
            .iter()
            .map(|e| e.name.clone())
            .collect();
        if !names.is_empty() {
            system.push_str(&format!("\n展厅展项：{}。", names.join("、")));
        }
        if let Some(idx) = exhibit_id
            .and_then(|id| self.theme.exhibits.iter().position(|e| e.id == id))
        {
            let e = &self.theme.exhibits[idx];
            system.push_str(&format!(
                "\n观众此刻站在「{}」展台前，已知资料：{}",
                e.name, e.deep
            ));
        }
        let user_text = if text.trim().is_empty() {
            "看看这张图，说说这是什么"
        } else {
            text
        };
        let msgs = vec![
            Message::system(system),
            Message::user(user_text).with_images(images.to_vec()),
        ];
        let raw = self.llm.chat(&msgs, 0.3, 220).ok()?;
        let cleaned = self.clean_llm(&raw, true);
        (!cleaned.is_empty()).then_some(cleaned)
    }

    fn llm_answer(
        &self,
        text: &str,
        ex: Option<&Exhibit>,
        temperature: f32,
        drop_redirect: bool,
    ) -> Option<String> {
        if !self.llm.available() {
            return None;
        }
        let mut msgs = vec![Message::system(self.system_prompt())];
        let content = match ex {
            Some(e) => format!(
                "（观众正站在「{}」展台前，已知资料：{}）{}",
                e.name, e.deep, text
            ),
            None => text.to_string(),
        };
        msgs.push(Message::user(content));
        let raw = self.llm.chat(&msgs, temperature, 220).ok()?;
        let cleaned = self.clean_llm(&raw, drop_redirect);
        (!cleaned.is_empty()).then_some(cleaned)
    }

    fn clean_llm(&self, text: &str, drop_redirect: bool) -> String {
        // Qwen3.5 等思考模型可能把 <think>…</think> 思维链一起吐出来。必须在去掉
        // markdown 符号之前剥离（md 正则含 `>`，先把标签削成 `<think` 就再也匹配不到了）。
        // 同时处理被长度截断、只有开头没有闭合的情况。
        let mut raw = text.to_string();
        loop {
            let Some(start) = raw.find("<think>") else { break };
            match raw[start..].find("</think>") {
                Some(rel_end) => {
                    let end = start + rel_end + "</think>".len();
                    raw.replace_range(start..end, "");
                }
                None => {
                    raw.truncate(start);
                    break;
                }
            }
        }
        let mut s = self.md.replace_all(&raw, "").to_string();
        s = s.lines().map(str::trim).collect::<Vec<_>>().join(" ");
        let mut s = s.trim().to_string();
        for p in ["小侨：", "小侨:", "讲解员：", "讲解员:"] {
            if let Some(rest) = s.strip_prefix(p) {
                s = rest.trim().to_string();
                break;
            }
        }
        if drop_redirect {
            s = self.trim_redirect(&s);
        }
        if char_len(&s) > 90 {
            let chars: Vec<char> = s.chars().collect();
            let mut cut = None;
            for (i, c) in chars.iter().enumerate().take(90) {
                if "。！？；".contains(*c) && i as f64 > 40.5 {
                    cut = Some(i + 1);
                }
            }
            s = match cut {
                Some(k) => chars[..k].iter().collect(),
                None => format!("{}。", chars[..90].iter().collect::<String>()),
            };
        }
        s.trim().to_string()
    }

    fn trim_redirect(&self, text: &str) -> String {
        let chars: Vec<char> = text.chars().collect();
        for i in (0..chars.len()).rev() {
            if "。！？".contains(chars[i]) {
                let mut j = i;
                while j > 0 && !"。！？".contains(chars[j - 1]) {
                    j -= 1;
                }
                let last: String = chars[j..=i].iter().collect();
                if self.redirect_hint.is_match(&last) && self.redirect_topic.is_match(&last) {
                    return chars[..j].iter().collect::<String>().trim().to_string();
                }
                break;
            }
        }
        text.to_string()
    }

    fn pack(
        &self,
        reply: String,
        action: &str,
        exhibit_idx: Option<usize>,
        emotion: Option<&str>,
    ) -> Answer {
        let ex = exhibit_idx.map(|i| &self.theme.exhibits[i]);
        let mut emo = emotion.map(str::to_string).unwrap_or_else(|| {
            match action {
                "welcome" | "chat" => "happy",
                _ => "neutral",
            }
            .to_string()
        });
        if emotion.is_none()
            && action == "explain"
            && (reply.contains('!') || reply.contains('！'))
        {
            emo = "happy".into();
        }
        let gesture = match action {
            "explain" => "point",
            "welcome" => "wave",
            _ => "idle",
        };
        Answer {
            reply,
            action: action.into(),
            emotion: emo,
            gesture: gesture.into(),
            exhibit: ex.map(|e| e.id.clone()),
            exhibit_name: ex.map(|e| e.name.clone()),
        }
    }
}
