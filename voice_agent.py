import asyncio, json
from datetime import datetime, timedelta, timezone
from langchain_core.tools import tool
from langchain_deepseek import ChatDeepSeek
from langchain.agents import create_agent
from livekit.agents import AgentSession, Agent, JobContext, WorkerOptions, cli, stt
from livekit.plugins import deepgram, groq, silero, langchain

import vector_store as vs
from job_tools import parse_query, get_job_details

def make_tools(room):
    async def ui(**msg):
        await room.local_participant.publish_data(json.dumps(msg), topic="ui")

    @tool
    async def search_jobs(query: str) -> str:
        """Search jobs from plain English, e.g. 'remote Python job over $50/hour this week'.
        Results are shown on the page automatically."""
        pq = parse_query(query)
        after = datetime.now(timezone.utc) - timedelta(days=pq.days) if pq.days else None
        hits = await asyncio.to_thread(
            vs.search_jobs, pq.semantic, limit=6,
            min_pay=pq.min_pay, max_pay=pq.max_pay, posted_after=after)
        # keep the payload small: LiveKit reliable data packets cap out around 15 KB
        slim = [{k: h[k] for k in ("job_id", "title", "pay", "skills", "url", "posted_at")} for h in hits]
        await ui(action="show_jobs", jobs=slim)
        return "\n".join(f"{h['job_id']}: {h['title']} ({h['pay']})" for h in hits) or "No matches."

    @tool
    async def open_job(job_id: str) -> str:
        """Open the full detail panel for one job by Job ID from search_jobs."""
        await ui(action="open_job", job_id=job_id)
        return await asyncio.to_thread(get_job_details, job_id)

    @tool
    async def highlight_apply(job_id: str) -> str:
        """Highlight the Apply button for a job (the user clicks it themselves)."""
        await ui(action="highlight_apply", job_id=job_id)
        return "Done."

    return [search_jobs, open_job, highlight_apply]

PROMPT = """You are a voice job-search assistant for AI-training and expert contractor jobs.
Reply in one to three short sentences. No markdown, lists or URLs. Say pay naturally
("fifty to a hundred dollars an hour"). Never say Job IDs aloud. Use search_jobs the
moment the user states what they want, and open_job when they ask about one result."""

def prewarm(proc):
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    await ctx.connect()
    vad = ctx.proc.userdata["vad"]

    graph = create_agent(
        ChatDeepSeek(model="deepseek-v4-flash", temperature=0),
        make_tools(ctx.room), prompt=PROMPT)

    session = AgentSession(
        vad=vad,
        stt=stt.FallbackAdapter(
            [deepgram.STT(model="nova-3"), groq.STT(model="whisper-large-v3-turbo")],
            vad=vad),
        llm=langchain.LLMAdapter(graph),
        tts=deepgram.TTS(model="aura-2-thalia-en"),
    )
    await session.start(agent=Agent(instructions=PROMPT), room=ctx.room)
    session.generate_reply(instructions="Greet the user and ask what kind of job they want.")

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm,
                              agent_name="jobs-agent", port=8081))