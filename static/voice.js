"use strict";
/*
 * The voice side of the page. Everything LiveKit-related lives here and nowhere else:
 *   1. get a token from /token (it also asks LiveKit to send the voice agent into the room)
 *   2. join the room, turn on the microphone, play the agent's audio (and show its avatar video if it has one)
 *   3. tell the orb (static/orb.js) what the agent is doing: connecting / listening / thinking / speaking
 *   4. forward every message the agent sends on topic "ui" to ONE method on window.SITE
 */
(() => {
  const btn = document.getElementById("voice-btn");
  const label = document.getElementById("voice-label");
  const dot = document.getElementById("voice-dot");
  const assistant = document.getElementById("assistant");
  const video = document.getElementById("avatar-video");
  const sound = document.getElementById("sound-btn");
  const IDLE_TEXT = "Talk to your job assistant";
  const LIVE_TEXT = "Live: just talk";
  // LiveKit's agent states -> the orb's states
  const ORB_STATE = { initializing: "connecting", idle: "listening", listening: "listening", thinking: "thinking", speaking: "speaking" };

  const orb = () => window.ORB;
  function visual(state) {
    assistant.dataset.state = state === "idle" || state === "connecting" ? "" : state; // drives the card's coloured border
    orb()?.setState(state);
  }

  const lk = window.LivekitClient;
  if (!lk) {
    label.textContent = "Voice unavailable (LiveKit failed to load)";
    btn.disabled = true;
    return;
  }
  const { Room, RoomEvent, Track } = lk;
  let room = null;

  async function start() {
    btn.disabled = true;
    label.textContent = "Connecting…";
    orb()?.unlock(); // browsers only allow audio analysis after a click, and this is the click
    visual("connecting");
    try {
      const res = await fetch("/token");
      if (!res.ok) {
        const detail = (await res.json().catch(() => ({}))).detail;
        throw new Error(detail || "Could not get a voice token");
      }
      const { url, token } = await res.json();

      const r = new Room();
      room = r;

      // The agent's voice (and avatar) arrive as tracks; they must be attached to elements to play.
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

      // initializing / listening / thinking / speaking
      r.on(RoomEvent.ParticipantAttributesChanged, (changed) => {
        const s = changed["lk.agent.state"];
        if (!s) return;
        visual(ORB_STATE[s] || "listening");
        label.textContent = LIVE_TEXT;
      });

      // Every tool call from the agent arrives here: { action, ...args } -> SITE[action](args)
      r.on(RoomEvent.DataReceived, (payload, participant, kind, topic) => {
        if (topic !== "ui") return;
        try {
          const { action, ...args } = JSON.parse(new TextDecoder().decode(payload));
          window.SITE?.[action]?.(args);
        } catch (err) {
          console.error("Bad message from the agent", err);
        }
      });

      // The dot lights up when LiveKit says it can hear you; the orb ripples faster while you speak.
      r.on(RoomEvent.ActiveSpeakersChanged, (speakers) => {
        const me = speakers.find((s) => s.isLocal);
        dot.classList.toggle("on", !!me);
        orb()?.userLevel(me ? Math.max(0.4, me.audioLevel || 0) : 0);
      });
      r.on(RoomEvent.Disconnected, stop);

      await r.connect(url, token);
      await r.localParticipant.setMicrophoneEnabled(true);
      btn.textContent = "Stop";
      btn.disabled = false;
      document.body.classList.add("voice-on");
      label.textContent = "Waiting for the assistant…";
    } catch (err) {
      await stop();
      label.textContent = err.message || "Could not start voice";
    }
  }

  async function stop() {
    const r = room;
    room = null; // clear first: disconnect() fires Disconnected, which calls stop() again
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
  }

  btn.addEventListener("click", () => (room ? stop() : start()));
  window.VOICE = { toggle: () => btn.click(), get active() { return !!room; } }; // the hero button in site.js uses this
})();