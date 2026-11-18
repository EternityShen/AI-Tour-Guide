// 音频：播放与口型同步 / 语音识别控制连接（麦克风由 Rust cpal 采集）
export class Speaker {
  constructor() {
    this.ctx = new AudioContext({ latencyHint: 'interactive' });
    this.sources = new Set();    // 所有在播的声音，切换问答时统一掐掉
    this._finishers = new Map(); // src -> finish()：stop 时主动结束 Promise（SUS-01）
  }

  async unlock() {
    if (this.ctx.state === 'suspended') await this.ctx.resume();
  }

  async play(audio, visemes, onViseme) {
    // 兼容两种入参：base64 字符串（短音频内联）或 ArrayBuffer（长音频走 /api/audio 拉取）
    let bytes;
    if (audio instanceof ArrayBuffer) {
      bytes = new Uint8Array(audio);
    } else {
      const bin = atob(String(audio || ''));
      bytes = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    }
    const audioBuf = await this.ctx.decodeAudioData(bytes.buffer);

    const src = this.ctx.createBufferSource();
    src.buffer = audioBuf;
    const analyser = this.ctx.createAnalyser();
    analyser.fftSize = 512;
    src.connect(analyser);
    analyser.connect(this.ctx.destination);

    const data = new Uint8Array(analyser.fftSize);
    const startAt = this.ctx.currentTime + 0.06;
    src.start(startAt);
    this.sources.add(src);

    let raf = 0;
    let done = false;
    let resolveFn = null;
    const finished = new Promise((resolve) => { resolveFn = resolve; });
    const finish = () => {
      if (done) return;
      done = true;
      cancelAnimationFrame(raf);
      this.sources.delete(src);
      this._finishers.delete(src);
      try { src.disconnect(); analyser.disconnect(); } catch (e) { /* 已断开 */ }
      onViseme(null, 0);
      resolveFn();
    };
    this._finishers.set(src, finish);
    src.onended = finish;

    const tick = () => {
      if (done) return;
      const t = this.ctx.currentTime - startAt;
      if (t >= 0 && t <= audioBuf.duration) {
        analyser.getByteTimeDomainData(data);
        let sum = 0;
        for (let i = 0; i < data.length; i++) {
          const v = (data[i] - 128) / 128;
          sum += v * v;
        }
        const rms = Math.sqrt(sum / data.length);
        const amp = Math.min(1, rms * 4.2);
        let cur = visemes[0] ? visemes[0].v : 'aa';
        for (const s of visemes) {
          if (t >= s.t && t < s.t + s.d) { cur = s.v; break; }
        }
        onViseme(cur, amp * 0.85);
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);

    return finished;
  }

  stop() {
    for (const s of Array.from(this.sources)) {
      try { s.stop(); } catch (e) { /* 已结束 */ }
      // 主动结束等待中的 play()：个别内核在 start 前停止或后台挂起时不派发 ended，
      // 否则 ask() 会永远 await 在播放上，界面卡在「讲解中」
      const f = this._finishers.get(s);
      if (f) f();
    }
    this.sources.clear();
    this._finishers.clear();
  }
}

// 麦克风采集已移到 Rust 端（cpal）：浏览器只负责建立 /ws/asr 控制连接，
// 不再申请 getUserMedia 权限，也不做 ScriptProcessor 降采样。
export class Mic {
  constructor() {
    this.ws = null;
    this.onText = null;
    this.onError = null;
    this.stopped = false;
    this.gotFinal = false;
  }

  async start(onPartial, onFinal, onError) {
    this.onText = { partial: onPartial, final: onFinal };
    this.onError = onError;
    this.stopped = false;
    this.gotFinal = false;

    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    const ws = new WebSocket(`${proto}://${location.host}/ws/asr`);
    this.ws = ws;
    // 语音服务没起来时不能无限等：5 s 内连不上就报错，让上层复位。
    // 连接建立后 cpal 才会打开麦克风，麦克风错误会作为 error 消息异步返回。
    await new Promise((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error('ASR_TIMEOUT')), 5000);
      ws.onopen = () => { clearTimeout(timer); resolve(); };
      ws.onerror = () => { clearTimeout(timer); reject(new Error('ASR_UNREACHABLE')); };
      ws.onclose = () => {
        if (ws.readyState !== WebSocket.OPEN) {
          clearTimeout(timer);
          reject(new Error('ASR_CLOSED'));
        }
      };
    });
    // 服务端在 final 后会主动关连接；已拿到 final 时不当作异常。
    ws.onclose = () => {
      if (!this.stopped && !this.gotFinal && this.onError) this.onError('语音服务连接已断开');
    };
    ws.onmessage = (ev) => {
      let d;
      try { d = JSON.parse(ev.data); } catch (e) { return; }
      if (d.type === 'partial') this.onText.partial(d.text);
      else if (d.type === 'final') { this.gotFinal = true; this.onText.final(d.text); }
      else if (d.type === 'error' && this.onError) this.onError(d.text);
    };
    ws.onerror = () => { if (!this.stopped && !this.gotFinal && this.onError) this.onError('语音服务连接失败'); };
  }

  async flush() {
    if (!this.ws || this.ws.readyState !== 1) return;
    // 文本帧 "END"：服务端停止 cpal 采集，把整段音频交给 valle-asr 出最终结果
    this.ws.send('END');
  }

  async stop() {
    this.stopped = true;
    const ws = this.ws;
    this.ws = null;
    if (!ws) return;
    // 先摘掉回调：close 是异步握手，超时兜底后仍在路上的 final/error
    // 若继续回调，会把上一轮的文本写进下一轮会话。
    ws.onmessage = null;
    ws.onerror = null;
    ws.onclose = null;
    try { ws.close(); } catch (e) { /* 忽略 */ }
  }
}
