"use strict";
/*
 * The voice side of the page. Everything LiveKit-related lives here and nowhere else:
 *   1. get a token from /token (it also asks LiveKit to send the voice agent into the room)
 *   2. join the room, turn on the microphone, play the agent's audio
 *   3. show what the agent is doing (listening / thinking / speaking)
 *   4. forward every message the agent sends on topic "ui" to ONE method on window.SITE
 */
(() => {
  const btn = document.getElementById("voice-btn");
  const label = document.getElementById("voice-label");
  const dot = document.getElementById("voice-dot");
  const IDLE_TEXT = "Talk to the assistant";
  const AGENT_STATES = {
    initializing: "Starting…",
    idle: "Ready",
    listening: "Listening…",
    thinking: "Thinking…",
    speaking: "Speaking…",
  };

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
    try {
      const res = await fetch("/token");
      if (!res.ok) {
        const detail = (await res.json().catch(() => ({}))).detail;
        throw new Error(detail || "Could not get a voice token");
      }
      const { url, token } = await res.json();

      const r = new Room();
      room = r;

      // The agent's voice arrives as an audio track; it must be attached to an element to play.
      r.on(RoomEvent.TrackSubscribed, (track) => {
        if (track.kind === Track.Kind.Audio) document.body.appendChild(track.attach());
      });
      r.on(RoomEvent.TrackUnsubscribed, (track) => track.detach().forEach((el) => el.remove()));

      // listening / thinking / speaking
      r.on(RoomEvent.ParticipantAttributesChanged, (changed) => {
        const s = changed["lk.agent.state"];
        if (s) label.textContent = AGENT_STATES[s] || s;
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

      // The dot lights up when LiveKit says it can hear you.
      r.on(RoomEvent.ActiveSpeakersChanged, (speakers) => dot.classList.toggle("on", speakers.some((s) => s.isLocal)));
      r.on(RoomEvent.Disconnected, stop);

      await r.connect(url, token);
      await r.localParticipant.setMicrophoneEnabled(true);
      btn.textContent = "Stop";
      btn.disabled = false;
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
    dot.classList.remove("on");
    btn.textContent = "Start";
    btn.disabled = false;
    label.textContent = IDLE_TEXT;
  }

  btn.addEventListener("click", () => (room ? stop() : start()));
})();