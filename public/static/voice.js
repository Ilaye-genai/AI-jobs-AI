"use strict";
/*
 * The voice side of the page. Everything LiveKit-related lives here and nowhere else:
 *   1. get a token from /token (it also asks LiveKit to send the voice agent into the room)
 *   2. join the room, turn on the microphone, play the agent's audio (and show its avatar video if it has one)
 *   3. tell the orb (static/orb.js) what the agent is doing: connecting / listening / thinking / speaking
 *   4. forward every message the agent sends on topic "ui" to ONE method on window.SITE
 *   5. SHOW EVERY PROBLEM in the assistant card (a red box with what failed, what usually causes it and the
 *      technical detail), so a failing deployment explains itself instead of loading forever. Problems come
 *      from this page (token, connection, microphone, the agent never joining) and from the agent itself
 *      (it sends {"action": "error", "text", "hint", "detail"} on topic "ui").
 */
(() => {
  const btn = document.getElementById("voice-btn");
  const label = document.getElementById("voice-label");
  const dot = document.getElementById("voice-dot");
  const assistant = document.getElementById("assistant");
  const video = document.getElementById("avatar-video");
  const sound = document.getElementById("sound-btn");

  // Drag Ella's floating card with the dedicated handle. Pointer events work for mouse and touch.
  const dragHandle = document.getElementById("assistant-drag");
  if (dragHandle) {
    let drag = null;
    const clamp = (value, min, max) => Math.min(Math.max(value, min), Math.max(min, max));

    // Restore the last position when the browser permits local storage.
    try {
      const saved = JSON.parse(localStorage.getItem("ella-assistant-position") || "null");
      if (saved && Number.isFinite(saved.x) && Number.isFinite(saved.y)) {
        const r = assistant.getBoundingClientRect();
        assistant.style.left = clamp(saved.x, 0, innerWidth - r.width) + "px";
        assistant.style.top = clamp(saved.y, 0, innerHeight - r.height) + "px";
        assistant.style.right = "auto";
        assistant.style.bottom = "auto";
      }
    } catch {}

    dragHandle.addEventListener("pointerdown", (event) => {
      if (event.button !== undefined && event.button !== 0) return;
      const rect = assistant.getBoundingClientRect();
      drag = { pointerId: event.pointerId, startX: event.clientX, startY: event.clientY,
        left: rect.left, top: rect.top, moved: false };
      dragHandle.setPointerCapture?.(event.pointerId);
      assistant.classList.add("dragging");
      event.preventDefault();
    });

    dragHandle.addEventListener("pointermove", (event) => {
      if (!drag || event.pointerId !== drag.pointerId) return;
      const dx = event.clientX - drag.startX;
      const dy = event.clientY - drag.startY;
      if (Math.abs(dx) + Math.abs(dy) > 3) drag.moved = true;
      const rect = assistant.getBoundingClientRect();
      const left = clamp(drag.left + dx, 0, innerWidth - rect.width);
      const top = clamp(drag.top + dy, 0, innerHeight - rect.height);
      assistant.style.left = left + "px";
      assistant.style.top = top + "px";
      assistant.style.right = "auto";
      assistant.style.bottom = "auto";
    });

    const finishDrag = (event) => {
      if (!drag || (event && event.pointerId !== drag.pointerId)) return;
      assistant.classList.remove("dragging");
      try {
        const rect = assistant.getBoundingClientRect();
        localStorage.setItem("ella-assistant-position", JSON.stringify({ x: rect.left, y: rect.top }));
      } catch {}
      drag = null;
    };
    dragHandle.addEventListener("pointerup", finishDrag);
    dragHandle.addEventListener("pointercancel", finishDrag);

    window.addEventListener("resize", () => {
      if (drag) return;
      const rect = assistant.getBoundingClientRect();
      const left = clamp(rect.left, 0, innerWidth - rect.width);
      const top = clamp(rect.top, 0, innerHeight - rect.height);
      assistant.style.left = left + "px";
      assistant.style.top = top + "px";
      assistant.style.right = "auto";
      assistant.style.bottom = "auto";
    });
  }
  const IDLE_TEXT = "Talk to your job assistant";
  const LIVE_TEXT = "Live: just talk";
  // LiveKit's agent states -> the orb's states
  const ORB_STATE = { initializing: "connecting", idle: "listening", listening: "listening", thinking: "thinking", speaking: "speaking" };

  const NO_AGENT_MS = 20000;  // nobody joined the room after this long: the agent is not running / not reachable
  const NO_START_MS = 60000;  // an agent joined but never reported a state after this long: it failed while starting

  // ------------------------------------------------------------------ the problem box
  const style = document.createElement("style");
  style.textContent = `
    #voice-problem { padding: 12px 14px; background: #3a1514; color: #ffd9d6; border-bottom: 1px solid #6b2522;
      font: 13px/1.45 system-ui, sans-serif; max-height: 45vh; overflow: auto; }
    #voice-problem .problem-item + .problem-item { margin-top: 10px; padding-top: 10px; border-top: 1px solid #6b2522; }
    #voice-problem strong { display: block; margin-bottom: 4px; color: #fff; font-size: 14px; }
    #voice-problem p { margin: 0 0 6px; }
    #voice-problem pre { margin: 6px 0; white-space: pre-wrap; word-break: break-word; opacity: .85;
      font: 11.5px/1.4 ui-monospace, Consolas, monospace; }
    #voice-problem button { font: inherit; background: transparent; color: inherit; border: 1px solid currentColor;
      border-radius: 999px; padding: 2px 10px; margin-right: 6px; cursor: pointer; }`;
  document.head.appendChild(style);

  const box = document.createElement("div");
  box.id = "voice-problem";
  box.hidden = true;
  box.setAttribute("role", "alert");
  const list = document.createElement("div"); // one block per problem
  const dismissBtn = document.createElement("button");
  const copyBtn = document.createElement("button");
  dismissBtn.type = copyBtn.type = "button";
  dismissBtn.textContent = "Dismiss";
  copyBtn.textContent = "Copy details";
  box.append(list, dismissBtn, copyBtn);
  assistant.prepend(box); // the card is anchored to the bottom of the screen, so the box grows upward

  const problems = []; // {title, hint, detail, kind}: several can be true at once (e.g. no database AND no avatar)
  const MAX_PROBLEMS = 4;

  function renderProblems() {
    list.replaceChildren(); // textContent only below, so text from the server or the agent can't inject HTML
    for (const p of problems) {
      const item = document.createElement("div");
      item.className = "problem-item";
      const t = document.createElement("strong");
      t.textContent = p.title;
      item.appendChild(t);
      if (p.hint) {
        const h = document.createElement("p");
        h.textContent = p.hint;
        item.appendChild(h);
      }
      if (p.detail) {
        const d = document.createElement("pre");
        d.textContent = p.detail;
        item.appendChild(d);
      }
      list.appendChild(item);
    }
    box.hidden = problems.length === 0;
    box.dataset.copy = problems.map((p) => [p.title, p.hint, p.detail].filter(Boolean).join("\n")).join("\n\n")
      + "\n\nPage: " + location.href + "\nTime: " + new Date().toISOString();
  }

  // Show a problem. The same title replaces its earlier copy; the newest MAX_PROBLEMS stay on screen.
  // `kind` lets a later event take back a warning that turned out to be temporary (see hideProblem).
  function problem(title, hint, detail, kind) {
    console.error("[voice] PROBLEM:", [title, hint, detail].filter(Boolean).join(" | "));
    const same = problems.findIndex((p) => p.title === title);
    if (same >= 0) problems.splice(same, 1);
    problems.push({ title, hint: hint || "", detail: detail || "", kind: kind || "" });
    while (problems.length > MAX_PROBLEMS) problems.shift();
    renderProblems();
    orb()?.endAction();
  }

  // With a kind: remove only those problems. Without: remove all of them.
  function hideProblem(kind) {
    for (let i = problems.length - 1; i >= 0; i--) if (!kind || problems[i].kind === kind) problems.splice(i, 1);
    renderProblems();
  }

  dismissBtn.onclick = () => hideProblem();
  copyBtn.onclick = async () => {
    try {
      await navigator.clipboard.writeText(box.dataset.copy || "");
      copyBtn.textContent = "Copied";
      setTimeout(() => (copyBtn.textContent = "Copy details"), 1500);
    } catch {} // clipboard can be blocked; the text is on screen anyway
  };

  // A failure that has already been shown: start()'s catch must not show a second, vaguer one.
  function fail(title, hint, detail, kind) {
    problem(title, hint, detail, kind);
    const e = new Error(title);
    e.shown = true;
    return e;
  }

  let timers = [];
  const later = (fn, ms) => timers.push(setTimeout(fn, ms));
  function clearTimers() {
    timers.forEach(clearTimeout);
    timers = [];
  }

  const orb = () => window.ORB;
  function visual(state) {
    assistant.dataset.state = state === "idle" || state === "connecting" ? "" : state; // drives the card's coloured border
    orb()?.setState(state);
  }

  const lk = window.LivekitClient;
  if (!lk) {
    label.textContent = "Voice unavailable (LiveKit failed to load)";
    problem("The LiveKit browser library did not load",
      "The page loads it from cdn.jsdelivr.net. An ad blocker, a firewall or a network problem is blocking it.",
      "Open the browser console (F12) and look for a blocked or failed request to livekit-client.");
    btn.disabled = true;
    return;
  }
  const { Room, RoomEvent, Track } = lk;
  let room = null;

  async function start() {
    btn.disabled = true;
    label.textContent = "Connecting…";
    hideProblem();
    clearTimers();
    orb()?.unlock(); // browsers only allow audio analysis after a click, and this is the click
    visual("connecting");
    try {
      // 1. a token from this website's own server
      let res;
      try {
        res = await fetch("/token");
      } catch (e) {
        throw fail("Could not reach this site's /token route",
          "The page could not contact its own server. Check your connection, and that the deployment is up.", String(e));
      }
      if (!res.ok) {
        const raw = await res.text().catch(() => "");
        let detail = raw;
        try { detail = JSON.parse(raw).detail || raw; } catch {}
        throw fail(`The server refused to give a voice token (HTTP ${res.status})`,
          res.status === 404
            ? "The /token route does not exist on this deployment. Open /api/health to see what the server can see."
            : "Missing LIVEKIT_URL, LIVEKIT_API_KEY or LIVEKIT_API_SECRET in the hosting settings is the usual cause. The server's own message is below.",
          String(detail).slice(0, 600));
      }
      const { url, token, room: roomName } = await res.json();
      let host = url;
      try { host = new URL(url).host; } catch {}
      const info = `LiveKit server: ${host}\nRoom: ${roomName}`;
      console.log("[voice] token ok.", info.replace("\n", " | "));

      const r = new Room();
      room = r;
      const agentState = () => {
        for (const p of r.remoteParticipants.values()) {
          const s = p.attributes && p.attributes["lk.agent.state"];
          if (s) return s;
        }
        return "";
      };

      // 2. the agent's voice (and avatar) arrive as tracks; they must be attached to elements to play
      r.on(RoomEvent.TrackSubscribed, (track) => {
        if (track.kind === Track.Kind.Audio) {
          document.body.appendChild(track.attach());
          orb()?.watchAudio(track.mediaStreamTrack); // the orb moves with the real voice
        } else if (track.kind === Track.Kind.Video) {
          track.attach(video);
          assistant.classList.add("has-video");
        }
      });
      r.on(RoomEvent.TrackUnsubscribed, (track) => {
        track.detach().forEach((el) => el !== video && el.remove());
        if (track.kind === Track.Kind.Video) assistant.classList.remove("has-video");
      });

      // Some browsers block audio until a tap; show a button for that.
      r.on(RoomEvent.AudioPlaybackStatusChanged, () => (sound.hidden = r.canPlaybackAudio));
      sound.onclick = () => r.startAudio().then(() => (sound.hidden = true));

      r.on(RoomEvent.ParticipantConnected, (p) => {
        console.log("[voice] joined the room:", p.identity);
        hideProblem("no-agent");
      });
      r.on(RoomEvent.Reconnecting, () => (label.textContent = "Connection lost, reconnecting…"));
      r.on(RoomEvent.Reconnected, () => (label.textContent = LIVE_TEXT));
      r.on(RoomEvent.MediaDevicesError, (e) =>
        problem("Microphone problem", "Allow microphone access for this site (padlock icon in the address bar), then press Start again.", String(e && e.message || e)));

      // initializing / listening / thinking / speaking
      r.on(RoomEvent.ParticipantAttributesChanged, (changed) => {
        const s = changed["lk.agent.state"];
        if (!s) return;
        hideProblem("no-agent");
        hideProblem("no-start");
        visual(ORB_STATE[s] || "listening");
        label.textContent = LIVE_TEXT;
      });

      // Every message from the agent arrives here: { action, ...args }. "error" is shown in the problem box,
      // everything else goes to SITE[action](args).
      r.on(RoomEvent.DataReceived, (payload, participant, kind, topic) => {
        if (topic !== "ui") return;
        try {
          const { action, ...args } = JSON.parse(new TextDecoder().decode(payload));
          if (action === "error") {
            problem(args.text || "The assistant reported a problem", args.hint || "This message came from the voice agent.", args.detail || "");
            return;
          }
          window.SITE?.[action]?.(args);
        } catch (err) {
          console.error("[voice] bad message from the agent", err);
        }
      });

      // The dot lights up when LiveKit says it can hear you; the orb ripples faster while you speak.
      r.on(RoomEvent.ActiveSpeakersChanged, (speakers) => {
        const me = speakers.find((s) => s.isLocal);
        dot.classList.toggle("on", !!me);
        orb()?.userLevel(me ? Math.max(0.4, me.audioLevel || 0) : 0);
      });

      // The connection ended without the user pressing Stop: say why.
      r.on(RoomEvent.Disconnected, (reason) => {
        if (room !== r) return; // the user pressed Stop
        const name = (lk.DisconnectReason && lk.DisconnectReason[reason]) || String(reason ?? "unknown");
        stop(true).then(() =>
          problem("The voice connection was closed", "LiveKit ended the session. Press Start to try again.", `Reason: ${name}\n${info}`));
      });

      // 3. join the room
      try {
        await r.connect(url, token);
      } catch (e) {
        throw fail("Could not join the LiveKit room",
          "Usually the LIVEKIT_URL, key or secret on the website is wrong, or the LiveKit project is paused or over its limit.",
          `${e && e.message || e}\n${info}`);
      }
      console.log("[voice] connected to the room");

      // 4. the microphone
      try {
        await r.localParticipant.setMicrophoneEnabled(true);
      } catch (e) {
        throw fail("The microphone could not be turned on",
          "Allow microphone access for this site (padlock icon in the address bar). Browsers only allow microphones on https:// pages and on localhost.",
          String(e && e.message || e));
      }

      btn.textContent = "Stop";
      btn.disabled = false;
      document.body.classList.add("voice-on");
      label.textContent = "Waiting for the assistant…";

      // 5. watchdogs: the two ways a deployment ends up "loading forever"
      later(() => {
        if (room === r && r.remoteParticipants.size === 0) {
          problem("No voice agent joined the room",
            "The website is working, but no agent answered. The agent is probably not running, or it is connected to a different LiveKit project, or it is registered under a different name (this site asks for the agent named in AGENT_NAME in server.py, \"jobs-agent\"). Look at the agent host's logs: a healthy agent logs \"registered worker\" when it starts and a job request when you press Start.",
            info, "no-agent");
        }
      }, NO_AGENT_MS);
      later(() => {
        if (room === r && r.remoteParticipants.size > 0 && !agentState()) {
          problem("The agent joined but never finished starting",
            "It entered the room and then stalled while starting up. Look at the agent host's logs for the first error: a missing or wrong API key (DEEPGRAM, DEEPSEEK, SYNTHESIA), the avatar failing to start, or the search models still loading.",
            info, "no-start");
        }
      }, NO_START_MS);
    } catch (err) {
      await stop(true);
      label.textContent = "Could not start voice";
      if (!err.shown) problem("Could not start voice", "An unexpected error. The details below say what failed.", String(err && err.message || err));
    }
  }

  // `keepProblem` is true when stopping because something failed: the problem box must stay on screen.
  async function stop(keepProblem) {
    const r = room;
    room = null; // clear first: disconnect() fires Disconnected, which calls this again
    clearTimers();
    if (r) await r.disconnect();
    document.querySelectorAll("audio").forEach((el) => el.remove());
    video.srcObject = null;
    assistant.classList.remove("has-video");
    sound.hidden = true;
    dot.classList.remove("on");
    document.body.classList.remove("voice-on");
    orb()?.userLevel(0);
    orb()?.endAction();
    visual("idle");
    btn.textContent = "Start";
    btn.disabled = false;
    label.textContent = IDLE_TEXT;
    if (keepProblem !== true) hideProblem();
  }

  btn.addEventListener("click", () => (room ? stop() : start()));
  window.VOICE = { toggle: () => btn.click(), get active() { return !!room; } }; // the hero button in site.js uses this
})();