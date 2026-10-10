"""
The agent's brain: a LangGraph agent (create_agent) running DeepSeek with the job tools.
No LiveKit in here, so you can test the thinking in a terminal before adding voice:

    python agent.py          # type to the agent; page updates are printed as [page <- {...}]

voice_agent.py imports build_agent() and plugs the same graph into the voice pipeline.
"""
import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_deepseek import ChatDeepSeek

from ui_tools import Send, make_tools

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

# Spoken aloud, so: short sentences, no markdown, no lists, no URLs, no IDs.
SYSTEM_PROMPT = """\
You are a friendly voice assistant that helps people find remote expert contract jobs
(mostly AI-training work) from a job board that sits on their screen next to this conversation.
You are in a spoken conversation, so answer in one to three short sentences. Never use markdown,
lists, URLs or Job IDs in what you say, and say pay naturally, for example "fifty to a hundred
dollars an hour".

How you work:
1. Learn what they want first: their field or skills, and a minimum pay or hours if they care.
   Ask at most two quick questions, one at a time. If they already told you enough, search at once.
2. Before you call search_jobs, say one short sentence such as "Let me look that up." so they are
   never left in silence. Put everything you know about their wishes into the query.
3. When results come back, name the best two in a sentence each, with their pay. Then call show_job
   for the single best match so the page opens it, and tell them you have opened it.
4. If they pick another job, call show_job for it. If they say they want to apply, call
   highlight_apply and tell them to press the Apply button themselves. You cannot apply for them.
5. If nothing matches, say so and suggest loosening pay, recency or field.

Rules:
- Only use jobs and Job IDs that your tools returned. Never invent a job, a pay rate or a requirement.
- If a job says pay is per task, tell the user the hourly figure is approximate.
- If they ask something unrelated to finding work, answer briefly and steer back."""


def build_agent(send: Send):
    """The compiled LangGraph graph. `send` delivers page updates to the user's browser."""
    model = ChatDeepSeek(model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"), temperature=0.3)
    return create_agent(model, make_tools(send), system_prompt=SYSTEM_PROMPT)


async def chat() -> None:
    """Terminal test: streams the reply token by token and prints every page update."""

    async def send(message: dict) -> None:
        print(f"\n   [page <- {message}]")

    agent = build_agent(send)
    messages: list = []
    print("Text test of the agent (no voice). Press Enter on an empty line to quit.\n")
    while True:
        text = (await asyncio.to_thread(input, "You: ")).strip()
        if not text:
            return
        messages.append({"role": "user", "content": text})
        print("Agent: ", end="", flush=True)
        async for mode, data in agent.astream({"messages": messages}, stream_mode=["messages", "values"]):
            if mode == "messages":
                chunk, _meta = data
                if type(chunk).__name__ == "AIMessageChunk" and isinstance(chunk.content, str):
                    print(chunk.content, end="", flush=True)
            else:  # "values": the full conversation so far, kept for the next turn
                messages = data["messages"]
        print("\n")


if __name__ == "__main__":
    asyncio.run(chat())