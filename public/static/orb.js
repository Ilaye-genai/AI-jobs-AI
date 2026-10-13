"use strict";
/*
 * The assistant's living orb.
 *
 * What it is: a small animation engine drawn on a <canvas> (no libraries, no build step) plus a
 * headline that always says what the assistant is doing right now.
 *
 *     idle        slow breathing, a few drifting sparks
 *     connecting  a bright arc circling the orb
 *     listening   soft ripples spreading outward (faster when YOU speak)
 *     thinking    three gold beads orbiting the orb
 *     speaking    a ring of voice bars driven by the assistant's real audio level
 *
 *   and short-lived ACTIONS the agent triggers while it works (they take over from the state):
 *     search      a radar sweep with job "blips" appearing
 *     found       a burst of blips            open / apply   expanding pulse rings
 *
 * How other files use it (all optional, nothing breaks if this file is missing):
 *     ORB.setState("listening")                    voice.js, from the agent's state
 *     ORB.action("Searching jobs…", "search")      site.js, when the agent sends a {"action": "status"} message
 *     ORB.endAction()                              the agent finished the action
 *     ORB.watchAudio(mediaStreamTrack)             voice.js, so the orb moves with the real voice
 *     ORB.userLevel(0..1)                          voice.js, how loud YOU are
 *     ORB.unlock()                                 voice.js, on the Start click (browsers need a click to use audio)
 *
 * The drawing code (class Engine) knows nothing about the page, so it can be tested on its own.
 */
(() => {
  const TAU = Math.PI * 2;
  const clamp = (v, a = 0, b = 1) => Math.min(b, Math.max(a, v));
  const lerp = (a, b, k) => a + (b - a) * k;
  const mix = (a, b, t) => [lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t)];
  const rgba = (c, a) => `rgba(${c[0] | 0},${c[1] | 0},${c[2] | 0},${clamp(a).toFixed(3)})`;

  // The palette matches the site: sage green, a brighter "listening" green, champagne gold for work.
  const C = { mint: [124, 196, 168], green: [78, 214, 148], gold: [232, 188, 108], ivory: [238, 236, 228] };

  // ---------------------------------------------------------------- presets
  // Every number glides towards its target, so changing state is a smooth morph, never a jump.
  //   amp    how much the edge wobbles          speed  how fast it moves        glow   halo strength
  //   scale  orb size                           rings / bars / beads / sweep / arc: which effect is on (0..1)
  const CHANNELS = ["amp", "speed", "glow", "scale", "rings", "bars", "beads", "sweep", "arc"];
  const full = (p) => ({ ...Object.fromEntries(CHANNELS.map((k) => [k, p[k] ?? 0])), color: p.color });

  const STATE = {
    idle:       { color: C.mint,  amp: 0.030, speed: 0.35, glow: 0.50, scale: 1.00 },
    connecting: { color: C.mint,  amp: 0.025, speed: 0.60, glow: 0.38, scale: 0.90, arc: 1 },
    listening:  { color: C.green, amp: 0.045, speed: 0.55, glow: 0.70, scale: 1.00, rings: 1 },
    thinking:   { color: C.gold,  amp: 0.055, speed: 1.10, glow: 0.65, scale: 0.90, beads: 1 },
    speaking:   { color: C.mint,  amp: 0.050, speed: 0.85, glow: 0.92, scale: 1.03, bars: 1 },
  };
  const KIND = {
    search: { color: C.gold,  amp: 0.045, speed: 1.20, glow: 0.78, scale: 0.92, sweep: 1 },
    found:  { color: C.green, amp: 0.050, speed: 0.90, glow: 0.95, scale: 1.02 },
    open:   { color: C.mint,  amp: 0.040, speed: 0.90, glow: 0.90, scale: 1.00 },
    apply:  { color: C.gold,  amp: 0.045, speed: 1.00, glow: 0.90, scale: 1.00 },
    work:   { color: C.gold,  amp: 0.055, speed: 1.10, glow: 0.70, scale: 0.90, beads: 1 },
  };

  // What the headline says when nobody gave it a text of their own.
  const STATE_COPY = {
    idle:       ["Tell me what you're looking for", "Press Start and just talk"],
    connecting: ["Connecting…", "Setting up the voice line"],
    listening:  ["I'm listening", "Say your field, pay or hours"],
    thinking:   ["Thinking…", "Working out the best match"],
    speaking:   ["Speaking", ""],
  };
  const KIND_COPY = {
    search: ["Searching jobs…", "Scanning the job board"],
    found:  ["Found your matches", "Best match first"],
    open:   ["Opening the job…", "Bringing it up on your screen"],
    apply:  ["Pointing at Apply…", "Watch for the pulsing button"],
    work:   ["Working on it…", ""],
  };

  // ================================================================ the engine (pure drawing + state)
  class Engine {
    constructor({ reduced = false } = {}) {
      this.reduced = reduced;
      this.state = "idle";
      this.kind = null;        // the current action, if any
      this.text = "";          // headline given by the agent for the current action
      this.sub = "";
      this.cur = full(STATE.idle);
      this.cur.color = [...C.mint];
      this.t = 0;              // animation time (slowed down for reduced motion)
      this.clock = 0;          // real time, for action timers
      this.minUntil = 0;       // an action stays visible at least this long...
      this.maxUntil = 0;       // ...and is dropped after this, in case the agent never says it finished
      this.pendingEnd = false;
      this.agentLevel = 0;     // measured loudness of the assistant's voice, 0..1 (set from outside)
      this.level = 0;          // the smoothed version that is actually drawn
      this.userLv = 0;         // how loud the user is
      this.userSmooth = 0;
      this.heard = false;      // did we get real audio since the assistant started speaking?
      this.ripples = [];
      this.blips = [];
      this.pulses = [];
      this.rippleClock = 0;
      this.blipClock = 0;
      this.sweepAngle = -1.2;
      this.sparks = Array.from({ length: 8 }, (_, i) => ({ a: i * 0.8, s: 0.10 + 0.045 * i, p: i * 1.7, k: 1.5 + (i % 3) * 0.32 }));
    }

    // ---- inputs
    setState(name) {
      if (!STATE[name] || name === this.state) return;
      this.state = name;
      this.heard = false;
      // The assistant has moved on to talking or waiting: the tool action is over (once it had its moment on screen).
      if (name === "speaking" || name === "listening") this.endAction();
    }

    action(kind, text = "", sub = "") {
      if (!KIND[kind]) kind = "work";
      this.kind = kind;
      this.text = text;
      this.sub = sub;
      this.pendingEnd = false;
      this.minUntil = this.clock + 1.4;
      this.maxUntil = this.clock + 14;
      if (kind === "open" || kind === "apply") {
        this.pulses.push({ born: this.t, color: kind === "apply" ? C.gold : C.mint });
        this.pulses.push({ born: this.t + 0.3, color: kind === "apply" ? C.gold : C.mint });
      }
      if (kind === "found") {
        for (let i = 0; i < 9; i++) this.blips.push({ born: this.t + i * 0.07, ang: -1.4 + i * 0.7 + Math.random() * 0.3, rad: 1.35 + Math.random() * 0.9, strong: 1 });
        this.pulses.push({ born: this.t, color: C.green });
      }
    }

    endAction() {
      if (this.kind) this.pendingEnd = true;
    }

    // What the headline should show right now.
    view() {
      if (this.kind) {
        const d = KIND_COPY[this.kind];
        return { key: `a:${this.kind}:${this.text}`, title: this.text || d[0], sub: this.sub || d[1], dots: this.kind !== "found" };
      }
      const c = STATE_COPY[this.state];
      return { key: `s:${this.state}`, title: c[0], sub: c[1], dots: this.state === "connecting" || this.state === "thinking" };
    }

    // ---- time
    step(dt) {
      dt = Math.min(dt, 0.05);
      this.clock += dt;
      this.t += dt * (this.reduced ? 0.25 : 1);

      if (this.kind && ((this.pendingEnd && this.clock >= this.minUntil) || this.clock >= this.maxUntil)) {
        this.kind = null;
        this.text = this.sub = "";
        this.pendingEnd = false;
      }

      const tg = full(this.kind ? KIND[this.kind] : STATE[this.state]);
      const k = 1 - Math.exp(-dt * 4.5);
      for (const key of CHANNELS) this.cur[key] += (tg[key] - this.cur[key]) * k;
      this.cur.color = mix(this.cur.color, tg.color, k);

      // The voice: fast attack, slow release, so the orb snaps to a syllable and relaxes after it.
      let want = 0;
      if (this.state === "speaking" && !this.kind) {
        if (this.agentLevel > 0.03) this.heard = true;
        // Until real audio shows up (or if the browser never lets us read it) imitate speech, so it never looks dead.
        const fake = 0.30 + 0.22 * Math.sin(this.t * 6.7) * Math.sin(this.t * 2.3 + 1);
        want = this.heard ? this.agentLevel : Math.max(this.agentLevel, fake);
      }
      this.level += (want - this.level) * (1 - Math.exp(-dt * (want > this.level ? 18 : 6)));
      this.userSmooth += (this.userLv - this.userSmooth) * (1 - Math.exp(-dt * 8));

      // listening ripples
      if (this.cur.rings > 0.05) {
        this.rippleClock += dt * (1 + this.userSmooth * 3);
        if (this.rippleClock > 1.5) {
          this.rippleClock = 0;
          this.ripples.push({ born: this.t, strength: 0.55 + this.userSmooth });
        }
      }
      this.ripples = this.ripples.filter((r) => this.t - r.born < 2.6);

      // search sweep + blips
      if (this.cur.sweep > 0.02) this.sweepAngle += dt * 2.3 * this.cur.sweep;
      if (this.cur.sweep > 0.5) {
        this.blipClock += dt;
        if (this.blipClock > 0.3) {
          this.blipClock = 0;
          this.blips.push({ born: this.t, ang: this.sweepAngle - 0.15, rad: 1.3 + Math.random() * 0.75, strong: 0 });
        }
      }
      this.blips = this.blips.filter((b) => this.t - b.born < 1.5);
      this.pulses = this.pulses.filter((p) => this.t - p.born < 1.2);
    }

    // ================================================================ drawing
    draw(ctx, W, H) {
      ctx.clearRect(0, 0, W, H);
      const m = this.cur;
      const t = this.t;
      const col = m.color;
      const cx = W / 2;
      const cy = H * 0.42;
      const base = Math.min(W * 0.5, H) * 0.25;
      const breathe = 1 + Math.sin(t * 1.25) * 0.018;
      const R = base * m.scale * breathe * (1 + this.level * 0.1);
      const amp = m.amp + this.level * 0.17;

      this.drawAura(ctx, cx, cy, R, col, m.glow);
      this.drawSweep(ctx, cx, cy, R, col, m.sweep);
      this.drawRipples(ctx, cx, cy, R, col, m.rings);
      this.drawBars(ctx, cx, cy, R, col, m.bars);
      this.drawBeads(ctx, cx, cy, R, col, m.beads, "behind");
      this.drawCore(ctx, cx, cy, R, col, amp, m.speed);
      this.drawBeads(ctx, cx, cy, R, col, m.beads, "front");
      this.drawArc(ctx, cx, cy, R, col, m.arc);
      this.drawBlips(ctx, cx, cy, R, col);
      this.drawPulses(ctx, cx, cy, R);
      this.drawSparks(ctx, cx, cy, R, col, m);
    }

    drawAura(ctx, cx, cy, R, col, glow) {
      const g = ctx.createRadialGradient(cx, cy, R * 0.55, cx, cy, R * 2.9);
      g.addColorStop(0, rgba(col, 0.34 * glow));
      g.addColorStop(0.42, rgba(col, 0.1 * glow));
      g.addColorStop(1, rgba(col, 0));
      ctx.fillStyle = g;
      ctx.fillRect(cx - R * 3, cy - R * 3, R * 6, R * 6);
    }

    blobPath(ctx, cx, cy, R, amp, speed, seed) {
      const N = 96;
      const t = this.t;
      ctx.beginPath();
      for (let i = 0; i <= N; i++) {
        const th = (i / N) * TAU;
        const w =
          Math.sin(th * 2 + t * speed + seed) +
          0.65 * Math.sin(th * 3 - t * speed * 1.35 + seed * 1.7) +
          0.4 * Math.sin(th * 5 + t * speed * 1.9 + seed * 2.3);
        const r = R * (1 + (amp * w) / 1.6);
        const x = cx + Math.cos(th) * r;
        const y = cy + Math.sin(th) * r;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.closePath();
    }

    drawCore(ctx, cx, cy, R, col, amp, speed) {
      const t = this.t;
      const light = mix(col, C.ivory, 0.55);
      const deep = mix(col, [8, 14, 11], 0.62);

      // body
      this.blobPath(ctx, cx, cy, R, amp, speed, 0);
      const body = ctx.createRadialGradient(cx - R * 0.3, cy - R * 0.36, R * 0.05, cx, cy, R * 1.12);
      body.addColorStop(0, rgba(light, 1));
      body.addColorStop(0.42, rgba(col, 0.97));
      body.addColorStop(1, rgba(deep, 1));
      ctx.fillStyle = body;
      ctx.fill();

      // liquid light moving inside the body
      ctx.save();
      this.blobPath(ctx, cx, cy, R, amp, speed, 0);
      ctx.clip();
      const swirls = [
        [cx + Math.cos(t * 0.7) * R * 0.38, cy + Math.sin(t * 0.9) * R * 0.32, R * 0.85, mix(col, C.ivory, 0.6), 0.34],
        [cx + Math.cos(t * 0.5 + 2) * R * 0.42, cy + Math.sin(t * 0.62 + 1) * R * 0.36, R * 0.75, mix(col, C.gold, 0.55), 0.24],
      ];
      for (const [x, y, r, c, a] of swirls) {
        const g = ctx.createRadialGradient(x, y, 0, x, y, r);
        g.addColorStop(0, rgba(c, a));
        g.addColorStop(1, rgba(c, 0));
        ctx.fillStyle = g;
        ctx.fillRect(cx - R * 1.4, cy - R * 1.4, R * 2.8, R * 2.8);
      }
      ctx.restore();

      // two translucent edge layers give the surface its shimmer
      ctx.save();
      ctx.globalCompositeOperation = "lighter";
      for (const [k, seed, a] of [[1.0, 2.1, 0.22], [0.96, 4.3, 0.14]]) {
        this.blobPath(ctx, cx, cy, R * k, amp * 1.25, speed * 1.15, seed);
        ctx.fillStyle = rgba(mix(col, C.ivory, 0.3), a);
        ctx.fill();
      }
      ctx.restore();

      // rim light
      this.blobPath(ctx, cx, cy, R, amp, speed, 0);
      ctx.strokeStyle = rgba(C.ivory, 0.22);
      ctx.lineWidth = 1.2;
      ctx.stroke();

      // specular highlight
      ctx.save();
      ctx.translate(cx - R * 0.3, cy - R * 0.4);
      ctx.rotate(-0.55);
      ctx.scale(1, 0.52);
      const h = ctx.createRadialGradient(0, 0, 0, 0, 0, R * 0.36);
      h.addColorStop(0, rgba(C.ivory, 0.5));
      h.addColorStop(1, rgba(C.ivory, 0));
      ctx.fillStyle = h;
      ctx.beginPath();
      ctx.arc(0, 0, R * 0.36, 0, TAU);
      ctx.fill();
      ctx.restore();
    }

    drawRipples(ctx, cx, cy, R, col, on) {
      if (on < 0.02) return;
      for (const r of this.ripples) {
        const p = (this.t - r.born) / 2.6;
        const a = Math.pow(1 - p, 1.7) * 0.5 * r.strength * on;
        ctx.beginPath();
        ctx.arc(cx, cy, R * (1.1 + 1.05 * (1 - Math.pow(1 - p, 2))), 0, TAU);
        ctx.strokeStyle = rgba(col, a);
        ctx.lineWidth = 1.6;
        ctx.stroke();
      }
    }

    drawBars(ctx, cx, cy, R, col, on) {
      if (on < 0.02) return;
      const M = 72;
      const t = this.t;
      const r0 = R * 1.2;
      ctx.lineCap = "round";
      ctx.lineWidth = 2;
      for (let i = 0; i < M; i++) {
        const th = (i / M) * TAU - Math.PI / 2;
        let s = 0.5 + 0.5 * Math.sin(i * 0.55 + t * 3.1);
        s *= 0.55 + 0.45 * Math.sin(i * 0.17 - t * 1.7 + 1);
        const len = R * (0.05 + 0.04 * s + 0.62 * this.level * s);
        ctx.beginPath();
        ctx.moveTo(cx + Math.cos(th) * r0, cy + Math.sin(th) * r0);
        ctx.lineTo(cx + Math.cos(th) * (r0 + len), cy + Math.sin(th) * (r0 + len));
        ctx.strokeStyle = rgba(mix(col, C.ivory, 0.25), (0.22 + 0.62 * s) * on);
        ctx.stroke();
      }
    }

    drawBeads(ctx, cx, cy, R, col, on, pass) {
      if (on < 0.02) return;
      const rx = R * 1.62;
      const ry = R * 0.56;
      const rot = -0.35;
      const cr = Math.cos(rot);
      const sr = Math.sin(rot);
      const at = (a) => [cx + Math.cos(a) * rx * cr - Math.sin(a) * ry * sr, cy + Math.cos(a) * rx * sr + Math.sin(a) * ry * cr];

      if (pass === "behind") {
        // the faint orbit path, seen once, behind the orb
        ctx.beginPath();
        for (let i = 0; i <= 64; i++) {
          const [x, y] = at((i / 64) * TAU);
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.strokeStyle = rgba(col, 0.12 * on);
        ctx.lineWidth = 1;
        ctx.stroke();
      }
      const bead = mix(col, C.ivory, 0.35);
      for (let b = 0; b < 3; b++) {
        for (let j = 0; j < 14; j++) {
          const a = this.t * 2.1 + (b * TAU) / 3 - j * 0.085;
          const front = Math.sin(a) >= 0;
          if ((pass === "front") !== front) continue;
          const [x, y] = at(a);
          const f = Math.pow(1 - j / 14, 1.6);
          ctx.beginPath();
          ctx.arc(x, y, (j === 0 ? 3.6 : 3.0) * (0.35 + 0.65 * f), 0, TAU);
          ctx.fillStyle = rgba(bead, 0.95 * f * on * (front ? 1 : 0.55));
          ctx.fill();
          if (j === 0) {
            const g = ctx.createRadialGradient(x, y, 0, x, y, 12);
            g.addColorStop(0, rgba(col, 0.5 * on));
            g.addColorStop(1, rgba(col, 0));
            ctx.fillStyle = g;
            ctx.fillRect(x - 12, y - 12, 24, 24);
          }
        }
      }
    }

    drawSweep(ctx, cx, cy, R, col, on) {
      if (on < 0.02) return;
      const rr = R * 2.1;
      for (const k of [1.45, 1.8, 2.1]) {
        ctx.beginPath();
        ctx.arc(cx, cy, R * k, 0, TAU);
        ctx.strokeStyle = rgba(col, 0.17 * on);
        ctx.lineWidth = 1;
        ctx.stroke();
      }
      const slices = 30;
      const span = 1.2;
      for (let i = 0; i < slices; i++) {
        const a1 = this.sweepAngle - (i * span) / slices;
        const a0 = a1 - span / slices - 0.004;
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.arc(cx, cy, rr, a0, a1);
        ctx.closePath();
        ctx.fillStyle = rgba(col, Math.pow(1 - i / slices, 1.6) * 0.5 * on);
        ctx.fill();
      }
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(cx + Math.cos(this.sweepAngle) * rr, cy + Math.sin(this.sweepAngle) * rr);
      ctx.strokeStyle = rgba(mix(col, C.ivory, 0.3), 0.6 * on);
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }

    drawBlips(ctx, cx, cy, R, col) {
      for (const b of this.blips) {
        const age = this.t - b.born;
        if (age < 0) continue;
        const life = Math.sin(Math.PI * clamp(age / 1.4));
        const x = cx + Math.cos(b.ang) * R * b.rad;
        const y = cy + Math.sin(b.ang) * R * b.rad;
        const size = 2.3 + (b.strong ? 1.6 : 0.6);
        const g = ctx.createRadialGradient(x, y, 0, x, y, size * 4.5);
        g.addColorStop(0, rgba(mix(col, C.ivory, 0.5), 0.85 * life));
        g.addColorStop(0.3, rgba(col, 0.35 * life));
        g.addColorStop(1, rgba(col, 0));
        ctx.fillStyle = g;
        ctx.fillRect(x - size * 4.5, y - size * 4.5, size * 9, size * 9);
        ctx.beginPath();
        ctx.arc(x, y, size, 0, TAU);
        ctx.fillStyle = rgba(C.ivory, 0.95 * life);
        ctx.fill();
      }
    }

    drawArc(ctx, cx, cy, R, col, on) {
      if (on < 0.02) return;
      const head = this.t * 2.6;
      ctx.lineCap = "round";
      ctx.lineWidth = 2.4;
      for (let k = 0; k < 12; k++) {
        ctx.beginPath();
        ctx.arc(cx, cy, R * 1.46, head - (k + 1) * 0.15, head - k * 0.15);
        ctx.strokeStyle = rgba(mix(col, C.ivory, 0.3), Math.pow(1 - k / 12, 1.4) * 0.85 * on);
        ctx.stroke();
      }
    }

    drawPulses(ctx, cx, cy, R) {
      for (const p of this.pulses) {
        const age = this.t - p.born;
        if (age < 0) continue;
        const q = clamp(age / 1.1);
        const ease = 1 - Math.pow(1 - q, 3);
        ctx.beginPath();
        ctx.arc(cx, cy, R * (1 + 1.55 * ease), 0, TAU);
        ctx.strokeStyle = rgba(mix(p.color, C.ivory, 0.25), Math.pow(1 - q, 1.5) * 0.9);
        ctx.lineWidth = 0.8 + 2.4 * (1 - q);
        ctx.stroke();
      }
    }

    drawSparks(ctx, cx, cy, R, col, m) {
      const calm = 1 - clamp(m.bars + m.beads + m.sweep);
      if (calm < 0.05) return;
      for (const s of this.sparks) {
        const a = s.a + this.t * s.s;
        const rr = R * (s.k + 0.18 * Math.sin(this.t * 0.4 + s.p));
        const x = cx + Math.cos(a) * rr;
        const y = cy + Math.sin(a) * rr * 0.72;
        const tw = 0.5 + 0.5 * Math.sin(this.t * 1.3 + s.p);
        const g = ctx.createRadialGradient(x, y, 0, x, y, 6);
        g.addColorStop(0, rgba(mix(col, C.ivory, 0.6), 0.75 * tw * calm * m.glow));
        g.addColorStop(1, rgba(col, 0));
        ctx.fillStyle = g;
        ctx.fillRect(x - 6, y - 6, 12, 12);
      }
    }
  }

  // Tests (node) use the engine directly.
  if (typeof module !== "undefined" && module.exports) module.exports = { Engine, C };
  if (typeof document === "undefined" || !document.getElementById) return;

  // ================================================================ page glue
  const canvas = document.getElementById("orb");
  const assistant = document.getElementById("assistant");
  const statusEl = document.getElementById("status");
  const titleEl = document.getElementById("status-title");
  const subEl = document.getElementById("status-sub");
  const reduced = window.matchMedia && matchMedia("(prefers-reduced-motion: reduce)").matches;
  const engine = new Engine({ reduced });
  const ctx = canvas && canvas.getContext ? canvas.getContext("2d") : null;

  // ---- audio: measure how loud the assistant's voice is, so the orb moves with it
  let actx = null;
  const taps = [];

  function unlock() {
    try {
      actx = actx || new (window.AudioContext || window.webkitAudioContext)();
      if (actx.state === "suspended") actx.resume();
    } catch {}
  }

  function watchAudio(mediaStreamTrack) {
    unlock();
    if (!actx || !mediaStreamTrack) return;
    try {
      const src = actx.createMediaStreamSource(new MediaStream([mediaStreamTrack]));
      const an = actx.createAnalyser();
      an.fftSize = 512;
      src.connect(an); // not connected to the speakers: the page already plays this track
      taps.push({ an, buf: new Uint8Array(an.fftSize), track: mediaStreamTrack });
    } catch (e) {
      console.warn("orb: could not read the assistant's audio level", e);
    }
  }

  function measure() {
    let peak = 0;
    for (let i = taps.length - 1; i >= 0; i--) {
      const tap = taps[i];
      if (tap.track.readyState === "ended") {
        taps.splice(i, 1);
        continue;
      }
      tap.an.getByteTimeDomainData(tap.buf);
      let sum = 0;
      for (const v of tap.buf) sum += ((v - 128) / 128) ** 2;
      peak = Math.max(peak, Math.sqrt(sum / tap.buf.length));
    }
    return clamp(peak * 3.4);
  }

  // ---- size the canvas to its box (sharp on high-DPI screens)
  let W = 0;
  let H = 0;
  function resize() {
    if (!canvas) return;
    const r = canvas.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    W = Math.max(1, r.width);
    H = Math.max(1, r.height);
    canvas.width = Math.round(W * dpr);
    canvas.height = Math.round(H * dpr);
    if (ctx) ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  if (canvas && window.ResizeObserver) new ResizeObserver(resize).observe(canvas);
  resize();

  // ---- the headline under the orb
  let lastKey = "";
  function paintText() {
    if (!titleEl) return;
    const v = engine.view();
    if (v.key === lastKey) return;
    lastKey = v.key;
    titleEl.classList.remove("swap");
    void titleEl.offsetWidth; // restart the fade-in
    titleEl.textContent = v.title.replace(/[….]+$/, ""); // the dots after it are animated by CSS
    titleEl.classList.add("swap");
    if (statusEl) statusEl.dataset.dots = v.dots ? "1" : "0";
    if (subEl) {
      subEl.textContent = v.sub || "";
      subEl.hidden = !v.sub;
    }
    if (statusEl) statusEl.dataset.kind = engine.kind || "";
  }

  // ---- the loop
  let last = performance.now();
  function frame(now) {
    const dt = (now - last) / 1000;
    last = now;
    if (!document.hidden) {
      if (engine.state === "speaking") engine.agentLevel = measure();
      engine.step(dt);
      paintText();
      const covered = assistant && assistant.classList.contains("has-video"); // the avatar video hides the orb
      if (ctx && !covered) engine.draw(ctx, W, H);
    }
    requestAnimationFrame(frame);
  }
  paintText();
  requestAnimationFrame(frame);

  window.ORB = {
    setState: (s) => engine.setState(s),
    action: (text, kind, sub) => {
      kind = kind || (/search|look/i.test(text) ? "search" : /apply/i.test(text) ? "apply" : /open|show/i.test(text) ? "open" : /found|match/i.test(text) ? "found" : "work");
      engine.action(kind, text, sub);
    },
    endAction: () => engine.endAction(),
    userLevel: (v) => (engine.userLv = clamp(v || 0)),
    watchAudio,
    unlock,
    engine, // handy in the browser console: ORB.engine.state, ORB.engine.kind
  };
})();