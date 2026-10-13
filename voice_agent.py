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

What the page shows while this runs (static/orb.js): the orb's state (connecting / listening / thinking /
speaking) comes from LiveKit by itself. The "status" messages sent below only cover the start-up steps that
LiveKit does not report; the tool actions ("Searching jobs…", "Found 6 jobs", ...) are sent from ui_tools.py.
A status carries {"text", "kind", "sub"}; kind is one of search / found / open / apply / work.
It ends by itself when the agent starts listening or speaking.

Problems are sent to the page too, so a failing deployment explains itself on screen (static/voice.js shows
them in a red box): {"action": "error", "text": <what failed>, "hint": <usual cause>, "detail": <technical>}.
"""
import asyncio
import dataclasses
import json
import logging
import os
import threading
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
GREETING = "Hello! I'm Ella, and I can help you find remote expert jobs. What kind of work are you looking for?"


_search_warm = threading.Event()


def _warm_up_search() -> None:
    """Load the embedding and reranker models now. Otherwise the user's FIRST search pays ~13 s for it,
    during which the voice pipeline stalls. Runs in a background thread; safe to call more than once."""
    if _search_warm.is_set():
        return
    _search_warm.set()
    try:
        import vector_store as vs

        vs.search_jobs("warm up", limit=1, candidates=1)
        logger.info("Search models are warm")
    except Exception as e:  # warming is only an optimisation
        logger.warning("Search warm-up skipped (%s)", e)


def prewarm(proc) -> None:
    proc.userdata["vad"] = silero.VAD.load()
    threading.Thread(target=_warm_up_search, daemon=True).start()


def make_groq_stt(vad):
    """The Groq Whisper backup, or None when there is no key or it cannot be built. Building the client
    makes an SSL context, which is slow on some Windows machines, so entrypoint runs this in a thread."""
    if not os.getenv("GROQ_API_KEY"):
        return None
    try:
        # Whisper is not a streaming model, so it is wrapped with VAD to behave like one.
        return stt.StreamAdapter(stt=groq.STT(model="whisper-large-v3-turbo", language="en"), vad=vad)
    except Exception as e:
        logger.warning("Groq Whisper backup not enabled (%s); using Deepgram only", e)
        return None


def build_stt(groq_stt):
    """Deepgram streams the speech; Groq Whisper takes over if Deepgram fails (when it was built)."""
    primary = deepgram.STT(model="nova-3", language="en")
    return stt.FallbackAdapter([primary, groq_stt]) if groq_stt else primary


AVATAR_START_TIMEOUT = 25.0  # seconds per attempt before giving up on the avatar and talking without it


def _short(err: BaseException, limit: int = 500) -> str:
    """One readable line about an exception, safe to put on the page (no secrets live in exception text here)."""
    text = f"{type(err).__name__}: {err}".strip()
    return text[:limit]


class StartupAbort(Exception):
    """The reason is already on the page; stop starting without adding a second, vaguer message."""


async def report(send, text: str, hint: str = "", detail: str = "") -> None:
    """Show a problem in the visitor's page (and in the log). Never raises."""
    logger.error("PROBLEM shown on the page: %s | %s | %s", text, hint, detail)
    try:
        await send({"action": "error", "text": text, "hint": hint, "detail": detail})
    except Exception as e:
        logger.warning("Could not send the problem to the page: %s", e)


async def check_settings(send) -> list[str]:
    """Report settings that are missing on THIS host (a deployed agent has no .env: secrets must be set in the
    host's settings). Returns the names that are missing and make the voice impossible."""
    needed = {
        "DEEPGRAM_API_KEY": "speech-to-text and the voice",
        "DEEPSEEK_API_KEY": "the assistant's thinking",
    }
    if os.getenv("USE_AVATAR", "1") != "0":
        needed["SYNTHESIA_API_KEY"] = "the avatar (without it, set USE_AVATAR=0 to run voice only)"
    missing = [k for k in needed if not os.getenv(k)]
    if missing:
        await report(
            send,
            "The agent host is missing settings",
            "Set these in the agent host's environment / secrets settings and restart it. A deployed agent does not read your .env file.",
            "\n".join(f"{k}: needed for {needed[k]}" for k in missing),
        )
    if not os.getenv("QDRANT_URL") and not Path(os.getenv("QDRANT_PATH", "qdrant_data")).exists():
        await report(
            send,
            "Job search has no database on the agent host",
            "QDRANT_URL is not set and there is no local qdrant_data folder, so every search will find nothing. Set QDRANT_URL and QDRANT_API_KEY (Qdrant Cloud) on the agent host.",
            "",
        )
    return missing


async def start_avatar(session: AgentSession, ctx: JobContext, send) -> bool:
    """
    Start the Synthesia avatar (it must start BEFORE session.start). Returns True when it is live.
    If it cannot start (no key, plan without Interactive Avatars, unknown avatar ID, timeout) the reason is
    logged AND shown on the page, and the call continues with voice only. Each attempt is capped at
    AVATAR_START_TIMEOUT seconds so a visitor is never left waiting minutes for a greeting.
    """
    if os.getenv("USE_AVATAR", "1") == "0":
        logger.info("USE_AVATAR=0: running without the avatar")
        return False

    avatar = synthesia.AvatarSession(synthesia.AvatarConfig(avatar_ids=[AVATAR_ID]), join_timeout=20.0)
    for attempt in range(2):  # cold starts occasionally time out, so retry once
        try:
            await asyncio.wait_for(avatar.start(session, room=ctx.room), timeout=AVATAR_START_TIMEOUT)
            logger.info("Synthesia avatar is live (%s)", AVATAR_ID)
            return True
        except asyncio.TimeoutError:
            if attempt == 0:
                logger.warning("Synthesia avatar did not start in %.0f s; retrying", AVATAR_START_TIMEOUT)
                continue
            await report(
                send,
                "The avatar did not start in time, continuing with voice only",
                "Synthesia did not join the room. Check SYNTHESIA_API_KEY, that your plan includes Interactive Avatars, and that the agent host can reach Synthesia.",
                f"Gave up after 2 attempts of {AVATAR_START_TIMEOUT:.0f} s. Avatar ID: {AVATAR_ID}",
            )
            return False
        except synthesia.SynthesiaError as e:
            if getattr(e, "retryable", False) and attempt == 0:
                logger.warning("Synthesia avatar not ready (%s); retrying", e)
                await asyncio.sleep(2)
                continue
            await report(
                send,
                "The avatar could not start, continuing with voice only",
                "FEATURE_NOT_IN_PLAN means your Synthesia plan lacks Interactive Avatars. UNKNOWN_AVATAR means the avatar ID is not in your workspace: set SYNTHESIA_AVATAR_ID.",
                f"type={getattr(e, 'type', None)} | {_short(e)}",
            )
            return False
        except Exception as e:  # anything else from the plugin (bad key, network): same treatment
            await report(
                send,
                "The avatar could not start, continuing with voice only",
                "Unexpected error from the avatar service. Check SYNTHESIA_API_KEY and the agent host's network access.",
                _short(e),
            )
            return False
    return False


async def entrypoint(ctx: JobContext) -> None:
    await ctx.connect()
    logger.info("Joined room %s", ctx.room.name)

    async def send(message: dict) -> None:
        """Tool calls use this to move the user's page (static/voice.js hands it to window.SITE)."""
        await ctx.room.local_participant.publish_data(json.dumps(message), topic="ui")

    try:
        await run_session(ctx, send)
    except StartupAbort as e:
        logger.error("Not starting the voice session: %s", e)
    except Exception as e:
        logger.exception("The voice session failed while starting")
        await report(
            send,
            "The assistant failed while starting",
            "The error below is the first thing that went wrong on the agent host. The agent's own log has the full trace.",
            _short(e),
        )
        raise


async def run_session(ctx: JobContext, send) -> None:
    # Start-up steps, shown big under the orb. They end by themselves once the agent starts listening/speaking.
    await send({"action": "status", "text": "Getting ready…", "kind": "work", "sub": "Loading the search models"})
    if await check_settings(send):
        raise StartupAbort("required settings are missing on the agent host")
    threading.Thread(target=_warm_up_search, daemon=True).start()  # no-op if prewarm already did it

    vad = ctx.proc.userdata["vad"]
    # Building these clients is slow on some machines; do it off the event loop, both at once.
    groq_stt, graph = await asyncio.gather(
        asyncio.to_thread(make_groq_stt, vad),
        asyncio.to_thread(build_agent, send),
    )
    session = AgentSession(
        vad=vad,
        stt=build_stt(groq_stt),
        llm=lk_langchain.LLMAdapter(graph=graph),  # streams the assistant's words to the voice as they arrive
        tts=deepgram.TTS(model="aura-2-asteria-en"),
        turn_handling={
            "turn_detection": "vad",                        # no extra turn-detector model to download
            "preemptive_generation": {"enabled": False},    # keep off when using tools
        },
    )

    # Errors raised later by the speech, language or voice services (bad key, quota, outage) land here.
    last_seen: dict[str, float] = {}

    def on_session_error(ev) -> None:
        err = getattr(ev, "error", ev)
        text = _short(err)
        now = asyncio.get_event_loop().time()
        if now - last_seen.get(text, -99.0) < 15:  # the same error repeats quickly: show it once
            return
        last_seen[text] = now
        asyncio.ensure_future(report(
            send,
            "A voice service reported an error",
            "Usually a wrong or expired API key, an exhausted quota, or a service outage (Deepgram, DeepSeek, Groq). The agent's log shows which.",
            text,
        ))

    session.on("error", on_session_error)

    if os.getenv("USE_AVATAR", "1") != "0":  # the avatar can take a while to join; say so
        await send({"action": "status", "text": "Starting the avatar…", "kind": "work", "sub": "This takes a few seconds"})
    avatar_live = await start_avatar(session, ctx, send)
    if not avatar_live:
        logger.info("Running voice only: the page shows the orb instead of the avatar")

    # instructions are empty on purpose: the system prompt lives inside the LangGraph agent
    await session.start(agent=Agent(instructions=""), room=ctx.room)
    logger.info("Session started; greeting the user")
    session.say(GREETING)  # the orb switches to "Speaking" by itself; no "ready" message is needed


def _worker_options() -> WorkerOptions:
    """Worker settings. A longer start-up allowance only when this LiveKit version supports it: the
    default 10 s is too short on a slow machine (the first process init timed out in your log)."""
    kwargs = dict(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm, agent_name=AGENT_NAME, port=8081)
    fields = {f.name for f in dataclasses.fields(WorkerOptions)}
    if "initialize_process_timeout" in fields:
        kwargs["initialize_process_timeout"] = 60.0
    if "num_idle_processes" in fields:
        kwargs["num_idle_processes"] = 1
    return WorkerOptions(**kwargs)


if __name__ == "__main__":
    cli.run_app(_worker_options())