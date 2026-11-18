use qx_core::Theme;

fn theme_path() -> String {
    // 项目重组后 themes/ 位于 qx-rust/ 下：qx-core 上两级即 qx-rust。
    concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../themes/qiaoxiang/theme.json"
    )
    .to_string()
}

#[test]
fn loads_and_validates_real_theme() {
    let t = Theme::load(theme_path()).expect("现有主题包应可解析");
    t.validate().expect("现有主题包应通过结构校验");
    assert_eq!(t.id, "qiaoxiang");
    assert_eq!(t.assistant.name, "小侨");
    assert!(!t.palette.accent.is_empty());
}

#[test]
fn exhibit_and_kb_counts_match_python_stats() {
    let t = Theme::load(theme_path()).unwrap();
    assert_eq!(t.exhibits.len(), 6, "对应 /api/stats 的 kb_exhibits");
    // kb_answers 口径 = sum(len(faq) + 2)：先按定义重算一遍，再钉住当前包的数量。
    let expect: usize = t.exhibits.iter().map(|e| e.faq.len() + 2).sum();
    assert_eq!(t.kb_answers(), expect, "kb_answers 口径应是 sum(len(faq)+2)");
    assert_eq!(t.kb_answers(), 32, "对应 /api/stats 的 kb_answers（6 展项 × 2 + 20 条问答）");
    for e in &t.exhibits {
        assert!(e.faq.len() >= 3, "每个展品至少 3 条 FAQ: {}", e.id);
        assert!(!e.script.is_empty(), "{} 应有讲解词", e.id);
        assert!(!e.deep.is_empty(), "{} 应深度讲解词", e.id);
        assert!(!e.aliases.is_empty(), "{} 应有别名", e.id);
    }
    // 互动问答的 followup 全靠问答本身命中（句子里没有展品别名），陈皮这几条是现场
    // 踩过的坑，少了就直接落进兜底话术。
    let chenpi = t.exhibit("chenpi").unwrap();
    assert!(chenpi.faq.iter().any(|f| f.q == "三年是什么意思"));
    assert!(chenpi.faq.iter().any(|f| f.q.contains("晒多久")));
    assert!(t.exhibit("diaolou").is_some());
    assert!(t.exhibit("chikan").is_some());
    assert!(t.exhibit("not_exist").is_none());
}

#[test]
fn rejects_incomplete_theme() {
    let empty = Theme::default();
    assert!(empty.validate().is_err(), "空主题包必须尽早失败（对应 DEF-14）");
}

#[test]
fn preserves_optional_render_fields() {
    let t = Theme::load(theme_path()).unwrap();
    let qiaopi = t.exhibit("qiaopi").expect("侨批银信应存在");
    assert!(qiaopi.model_scale.is_some(), "modelScale 应被 camelCase 映射");
    assert!(qiaopi.plaque.is_some());
    let diaolou = t.exhibit("diaolou").unwrap();
    assert!(diaolou.materials.is_some(), "材质覆盖表应透传");
}
