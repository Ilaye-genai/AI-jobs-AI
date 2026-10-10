"""
The voice side: LiveKit wiring only. The thinking lives in agent.py, the page actions in ui_tools.py.

    mic -> VAD -> Deepgram STT (Groq Whisper as backup) -> LangGraph + DeepSeek -> Deepgram TTS
                                                                  |                     |
                                                       tool calls -> page          Synthesia avatar -> browser

Run (two terminals):
    python server.py             # the website, http://localhost:8080
    python voice_agent.py dev    # this agent; it joins the room the page creates when you press Start

.env:  LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET, DEEPGRAM_API_KEY, DEEPSEEK_API_KEY,
       SYNTHESIA_API_KEY (needs a plan with Interactive Avatars), GROQ_API_KEY (optional backup STT)
       SYNTHESIA_AVATAR_ID (optional), USE_AVATAR=0 (optional: voice only, to isolate problems)
"""
import asyncio
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli, stt
from livekit.plugins import deepgram, groq, silero, synthesia
from livekit.plugins import langchain as lk_langchain

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

from agent import build_agent  # noqa: E402  (after load_dotenv so the DeepSeek key is set)

logger = logging.getLogger("jobs-agent")

AGENT_NAME = "jobs-agent"  # must match AGENT_NAME in server.py
# The stock avatar Tim's demo uses ("Jenny"). If you get UNKNOWN_AVATAR it is not in your workspace's
# gallery: put the ID of one that is in SYNTHESIA_AVATAR_ID in .env.
DEFAULT_AVATAR_ID = "8788bef1-8020-46e0-a8f4-510ea9989b25"
AVATAR_ID = os.getenv("SYNTHESIA_AVATAR_ID") or DEFAULT_AVATAR_ID
GREETING = "Hello! I am Ella I can help you find remote expert jobs. What kind of work are you looking for?"


def prewarm(proc) -> None:
    proc.userdata["vad"] = silero.VAD.load()


def build_stt(vad):
    """Deepgram streams the speech; Groq Whisper takes over if Deepgram fails (when a Groq key is set)."""
    primary = deepgram.STT(model="nova-3", language="en")
    if not os.getenv("GROQ_API_KEY"):
        return primary
    try:
        # Whisper is not a streaming model, so it is wrapped with VAD to behave like one.
        backup = stt.StreamAdapter(stt=groq.STT(model="whisper-large-v3-turbo", language="en"), vad=vad)
        return stt.FallbackAdapter([primary, backup])
    except Exception as e:
        logger.warning("Groq Whisper backup not enabled (%s); using Deepgram only", e)
        return primary


async def start_avatar(session: AgentSession, ctx: JobContext) -> bool:
    """
    Start the Synthesia avatar (it must start BEFORE session.start). Returns True when it is live.
    If it cannot start (no key, plan without Interactive Avatars, unknown avatar ID, timeout) the
    reason is logged loudly and the call continues with voice only, so you can still test the rest.
    """
    if os.getenv("USE_AVATAR", "1") == "0":
        logger.info("USE_AVATAR=0: running without the avatar")
        return False

    avatar = synthesia.AvatarSession(synthesia.AvatarConfig(avatar_ids=[AVATAR_ID]), join_timeout=60.0)
    for attempt in range(3):  # cold starts occasionally time out, so retry
        try:
            await avatar.start(session, room=ctx.room)
            logger.info("Synthesia avatar is live (%s)", AVATAR_ID)
            return True
        except synthesia.SynthesiaError as e:
            if e.retryable and attempt < 2:
                logger.warning("Synthesia avatar not ready (%s); retrying", e)
                await asyncio.sleep(2)
                continue
            logger.error(
                "SYNTHESIA AVATAR FAILED (type=%s): %s. Continuing with voice only. "
                "FEATURE_NOT_IN_PLAN = your plan lacks Interactive Avatars; UNKNOWN_AVATAR = set SYNTHESIA_AVATAR_ID.",
                getattr(e, "type", None), e,
            )
            return False
    return False


async def entrypoint(ctx: JobContext) -> None:
    await ctx.connect()
    logger.info("Joined room %s", ctx.room.name)

    async def send(message: dict) -> None:
        """Tool calls use this to move the user's page (static/voice.js hands it to window.SITE)."""
        await ctx.room.local_participant.publish_data(json.dumps(message), topic="ui")

    vad = ctx.proc.userdata["vad"]
    session = AgentSession(
        vad=vad,
        stt=build_stt(vad),
        llm=lk_langchain.LLMAdapter(graph=build_agent(send)),  # streams tokens to the voice as they arrive
        tts=deepgram.TTS(model="aura-2-asteria-en"),
        turn_handling={
            "turn_detection": "vad",                        # no extra turn-detector model to download
            "preemptive_generation": {"enabled": False},    # keep off when using tools
        },
    )

    avatar_live = await start_avatar(session, ctx)
    if not avatar_live:
        await send({"action": "status", "text": "Voice only (avatar unavailable)"})

    # instructions are empty on purpose: the system prompt lives inside the LangGraph agent
    await session.start(agent=Agent(instructions=""), room=ctx.room)
    logger.info("Session started; greeting the user")
    session.say(GREETING)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm, agent_name=AGENT_NAME, port=8081))