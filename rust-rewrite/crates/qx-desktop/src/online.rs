//! 联网实时信息：天气 / 新闻 / 比分。
//!
//! 设计约束（与项目整体一致）：
//! 1. 不引入 TLS 依赖 —— 走 Windows 自带 `curl.exe`（Win10 1803+ 起内置），
//!    单次请求 2 秒硬超时，失败即返回 None，由上层退回原有「查不到」话术；
//! 2. 只在识别到明确意图时才发请求，不打扰离线主线；
//! 3. 结果直接成句，不经大模型改写 —— 天气与比分不需要创作，模板更稳更快，
//!    也不会把小模型编造的数字混进来；
//! 4. 断网、无 curl、接口改版，一律静默失败，绝不阻塞对话。

use serde_json::Value;
use std::process::{Command, Stdio};

/// 江门市中心坐标，天气默认落点。
const JIANGMEN: (f64, f64) = (22.578, 113.081);

/// 识别到的实时意图。
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Intent {
    Weather,
    News,
    Score,
}

/// 关键词判定。宁可漏判（退回原话术），不可错判（把展厅讲解打断成天气播报）。
pub fn detect(text: &str) -> Option<Intent> {
    let t = text.trim();
    // 超过 24 字的多半是「顺便问一句天气，然后继续问展项」的混合长句，
    // 交给大模型顺着聊更自然，这里不抢答。
    if t.chars().count() > 24 {
        return None;
    }
    const WEATHER: &[&str] = &[
        "天气", "气温", "下雨", "会不会雨", "几度", "冷不冷", "热不热", "带伞", "台风",
    ];
    const NEWS: &[&str] = &["新闻", "今天发生", "最近发生", "时事", "头条", "热搜"];
    const SCORE: &[&str] = &["比分", "球赛", "比赛结果", "谁赢了", "几比几", "NBA", "足球赛"];

    if WEATHER.iter().any(|k| t.contains(k)) {
        return Some(Intent::Weather);
    }
    if SCORE.iter().any(|k| t.contains(k)) {
        return Some(Intent::Score);
    }
    if NEWS.iter().any(|k| t.contains(k)) {
        return Some(Intent::News);
    }
    None
}

/// HTTP GET：优先 curl.exe，2 秒超时，最多 512 KB。
fn http_get(url: &str) -> Option<String> {
    let curl = r"C:\Windows\System32\curl.exe";
    if !std::path::Path::new(curl).exists() {
        return None;
    }
    let out = Command::new(curl)
        .args([
            "-sS",
            // 冷启动的 TLS 握手约 1–2 秒，总超时给 2 秒会把握手本身卡掉
            // （实测 2.44 秒失败＝正好撞上 2 秒上限）。连接 2 秒、总计 4 秒。
            "--connect-timeout",
            "2",
            "--max-time",
            "4",
            "--compressed",
            "-A",
            "Mozilla/5.0 (QiaoxiangGuide)",
            "-w",
            "\n%{http_code}",
        ])
        .arg(url)
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let raw = String::from_utf8_lossy(&out.stdout).to_string();
    let (body, code) = raw.rsplit_once('\n')?;
    if code.trim() != "200" {
        return None;
    }
    if body.len() > 512 * 1024 {
        return None;
    }
    Some(body.to_string())
}

/// WMO 天气代码 → 中文。
fn wmo(code: i64) -> &'static str {
    match code {
        0 => "晴",
        1 => "晴间多云",
        2 => "多云",
        3 => "阴",
        45 | 48 => "有雾",
        51 | 53 | 55 => "毛毛雨",
        56 | 57 => "冻雨",
        61 => "小雨",
        63 => "中雨",
        65 => "大雨",
        66 | 67 => "冻雨",
        71 | 73 | 75 => "下雪",
        77 => "雪粒",
        80 | 81 => "阵雨",
        82 => "强阵雨",
        85 | 86 => "阵雪",
        95 => "雷阵雨",
        96 | 99 => "雷暴伴冰雹",
        _ => "天气多变",
    }
}

fn weather() -> Option<String> {
    let url = format!(
        "https://api.open-meteo.com/v1/forecast?latitude={}&longitude={}\
         &current=temperature_2m,relative_humidity_2m,weather_code&timezone=Asia%2FShanghai",
        JIANGMEN.0, JIANGMEN.1
    );
    let body = http_get(&url)?;
    let v: Value = serde_json::from_str(&body).ok()?;
    let cur = v.get("current")?;
    let temp = cur.get("temperature_2m")?.as_f64()?;
    let rh = cur.get("relative_humidity_2m")?.as_i64().unwrap_or(0);
    let code = cur.get("weather_code")?.as_i64().unwrap_or(-1);
    Some(format!(
        "江门现在 {:.0}°C，{}，湿度 {}%。这个天在展厅里走一圈正舒服——想先看哪一件？",
        temp,
        wmo(code),
        rh
    ))
}

/// 去掉 RSS 里的标签与实体，取纯文本标题。
///
/// CDATA 必须先脱壳再去标签：`<![CDATA[标题]]>` 里的 `<` 会被当成标签起点，
/// 一路吃到 `]]>`，把整段标题吞掉（这个 bug 由单测抓出来）。
fn strip_tags(s: &str) -> String {
    let s = s.replace("<![CDATA[", "").replace("]]>", "");
    let mut out = String::new();
    let mut in_tag = false;
    for ch in s.chars() {
        match ch {
            '<' => in_tag = true,
            '>' => in_tag = false,
            c if !in_tag => out.push(c),
            _ => {}
        }
    }
    out.replace("&quot;", "\"")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&nbsp;", " ")
        .replace("&#8195;", " ")
        .trim()
        .to_string()
}

/// 只取 `<item>` 里的 `<title>`，频道标题天然被排除，不必靠「跳过第一条」猜。
fn titles_of(xml: &str, limit: usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = xml;
    while out.len() < limit {
        let Some(i) = rest.find("<item>") else { break };
        let after_item = &rest[i + 6..];
        let Some(end) = after_item.find("</item>") else { break };
        let block = &after_item[..end];
        if let Some(ti) = block.find("<title>") {
            let after_t = &block[ti + 7..];
            if let Some(tj) = after_t.find("</title>") {
                let t = strip_tags(&after_t[..tj]);
                if t.chars().count() >= 4 && !out.contains(&t) {
                    out.push(t);
                }
            }
        }
        rest = &after_item[end + 7..];
    }
    out
}

fn news() -> Option<String> {
    let body = http_get("https://www.chinanews.com.cn/rss/scroll-news.xml")?;
    let all = titles_of(&body, 2);
    if all.len() < 2 {
        return None;
    }
    let picks: Vec<String> = all
        .into_iter()
        .take(2)
        .map(|mut s| {
            if s.chars().count() > 34 {
                s = s.chars().take(34).collect::<String>() + "…";
            }
            s
        })
        .collect();
    Some(format!(
        "刚看到两条：{}；{}。这些离侨乡远了些，不过馆里那几件东西，故事还长着。",
        picks[0], picks[1]
    ))
}

fn score() -> Option<String> {
    let body = http_get("https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard")?;
    let v: Value = serde_json::from_str(&body).ok()?;
    let events = v.get("events")?.as_array()?;
    let mut lines = Vec::new();
    for ev in events.iter().take(2) {
        let name = ev.get("name").and_then(|x| x.as_str()).unwrap_or("");
        let status = ev
            .get("status")
            .and_then(|s| s.get("type"))
            .and_then(|t| t.get("description"))
            .and_then(|d| d.as_str())
            .unwrap_or("");
        let comps = ev.get("competitions")?.as_array()?.first()?;
        let teams = comps.get("competitors")?.as_array()?;
        let mut scores = Vec::new();
        for c in teams.iter().take(2) {
            let team = c
                .get("team")
                .and_then(|t| t.get("abbreviation"))
                .and_then(|t| t.as_str())
                .unwrap_or("?");
            let sc = c.get("score").and_then(|s| s.as_str()).unwrap_or("-");
            scores.push(format!("{team} {sc}"));
        }
        if scores.len() == 2 {
            lines.push(format!("{}（{}）", scores.join(" - "), status));
        } else if !name.is_empty() {
            lines.push(name.to_string());
        }
    }
    if lines.is_empty() {
        return None;
    }
    Some(format!(
        "NBA 目前：{}。看球是看球，碉楼的故事也不亏——要接着听吗？",
        lines.join("；")
    ))
}

/// 对外入口：命中意图且取数成功才返回成句结果。
pub fn answer(text: &str) -> Option<String> {
    match detect(text)? {
        Intent::Weather => weather(),
        Intent::News => news(),
        Intent::Score => score(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_weather() {
        assert_eq!(detect("江门今天天气怎么样"), Some(Intent::Weather));
        assert_eq!(detect("外面下雨吗"), Some(Intent::Weather));
    }

    #[test]
    fn detect_news_and_score() {
        assert_eq!(detect("今天有什么新闻"), Some(Intent::News));
        assert_eq!(detect("昨天比赛比分多少"), Some(Intent::Score));
    }

    #[test]
    fn detect_ignores_exhibit_questions() {
        assert_eq!(detect("开平碉楼为什么叫碉楼"), None);
        assert_eq!(detect("陈皮有什么功效"), None);
        // 长句不做实时查询
        assert_eq!(
            detect("我想知道今天江门的天气怎么样以及展厅里第一件展品是什么"),
            None
        );
    }

    #[test]
    fn strip_tags_plain() {
        assert_eq!(strip_tags("<![CDATA[标题]]>"), "标题");
        assert_eq!(strip_tags("  a &amp; b "), "a & b");
    }

    #[test]
    fn titles_parse() {
        let xml = "<rss><channel><title>中新网即时新闻</title>\
                   <item><title>美国费城交响乐团中国巡演启幕</title></item>\
                   <item><title><![CDATA[中国海油取得超深层油气勘探突破]]></title></item>\
                   </channel></rss>";
        let t = titles_of(xml, 2);
        assert_eq!(t.len(), 2);
        assert_eq!(t[0], "美国费城交响乐团中国巡演启幕");
        // CDATA 必须脱壳成功，否则这条会变成空串
        assert_eq!(t[1], "中国海油取得超深层油气勘探突破");
    }

    #[test]
    fn wmo_covers_common() {
        assert_eq!(wmo(0), "晴");
        assert_eq!(wmo(61), "小雨");
        assert_eq!(wmo(999), "天气多变");
    }

    /// 真实联网用例：`cargo test -p qx-desktop live_online -- --ignored --nocapture`
    /// 默认不跑，避免无网环境下 CI/本地测试变红。
    #[test]
    #[ignore = "需要联网，手动执行"]
    fn live_online() {
        let w = answer("江门今天天气怎么样");
        println!("weather => {w:?}");
        assert!(w.is_some(), "天气取数失败");

        let n = answer("今天有什么新闻");
        println!("news => {n:?}");
        assert!(n.is_some(), "新闻取数失败");

        let s = answer("昨天NBA比分");
        println!("score => {s:?}");
        assert!(s.is_some(), "比分取数失败");

        // 展项问题绝不能走实时查询
        assert!(answer("开平碉楼为什么叫碉楼").is_none());
    }
}
