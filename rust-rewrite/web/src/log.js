// 前端运行日志：未捕获异常 / 未处理的 Promise 拒绝 / 资源加载失败回传到本机
// /api/log（仅 127.0.0.1，不经过外网）。日志失败必须静默，绝不影响讲解主流程。
(() => {
  const ENDPOINT = '/api/log';
  const WINDOW_MS = 60000;
  const MAX_PER_WINDOW = 30;
  const DEDUPE_MS = 10000;

  let sent = 0;
  let windowStart = Date.now();
  const recent = new Map();

  const cut = (v, n) => (typeof v === 'string' ? v.slice(0, n) : '');

  function allow(key) {
    const now = Date.now();
    if (now - windowStart > WINDOW_MS) {
      windowStart = now;
      sent = 0;
      recent.clear();
    }
    if (sent >= MAX_PER_WINDOW) return false;
    if (recent.has(key) && now - recent.get(key) < DEDUPE_MS) return false;
    recent.set(key, now);
    sent += 1;
    return true;
  }

  function send(level, message, extra) {
    try {
      const info = extra || {};
      const msg = cut(String(message == null ? '' : message), 1800);
      const key = level + '|' + cut(msg, 120);
      if (!allow(key)) return;

      const payload = JSON.stringify({
        level: level || 'error',
        message: msg,
        source: cut(String(info.source || ''), 120),
        stack: cut(String(info.stack || ''), 4000),
        url: cut(String(location && location.href), 300),
        line: Number(info.line) || 0,
        col: Number(info.col) || 0,
        ts: new Date().toISOString(),
      });

      if (navigator.sendBeacon) {
        const blob = new Blob([payload], { type: 'application/json' });
        if (navigator.sendBeacon(ENDPOINT, blob)) return;
      }
      fetch(ENDPOINT, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: payload,
        keepalive: true,
      }).catch(() => {});
    } catch (_) {
      // 日志自身的异常不能冒泡到页面
    }
  }

  window.qxLog = send;

  window.addEventListener('error', (e) => {
    // 资源加载失败时没有 e.message，target 是 img/script/link 等元素
    if (e.target && e.target !== window) {
      const src = e.target.src || e.target.href || e.target.tagName || 'resource';
      send('error', '资源加载失败：' + src, { source: 'resource' });
      return;
    }
    send('error', e.message || '未捕获异常', {
      source: e.filename || 'window.onerror',
      stack: e.error && e.error.stack,
      line: e.lineno,
      col: e.colno,
    });
  }, true);

  window.addEventListener('unhandledrejection', (e) => {
    const r = e.reason;
    send('error', '未处理的 Promise 拒绝：' + ((r && r.message) || r || ''), {
      source: 'unhandledrejection',
      stack: r && r.stack,
    });
  });
})();
