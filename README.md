# AI Jobs AI

A full-stack AI job board and voice assistant for finding remote expert-contract and AI-training jobs. The app scrapes public job listings, normalizes and enriches them, stores them in a searchable index, and exposes both a browser-based job board and a voice-enabled agent that can help a user find relevant opportunities.

This project combines:
- a Python job scraper and normalizer
- a Qdrant-backed semantic search layer
- a FastAPI job API and browser UI
- a LangGraph/DeepSeek-powered agent
- a LiveKit voice assistant for spoken interaction

The job data file itself is intentionally treated as the project’s source-of-truth dataset and is not described in detail here.

## Project goals

- Collect AI and remote expert-work jobs from public sources
- Normalize titles, pay, locations, skills, and descriptions
- Make the jobs searchable by semantics and by practical filters like pay and recency
- Present them in a clean browser board
- Give a voice assistant the ability to search, open, and recommend jobs conversationally

## High-level architecture

The project uses a simple but effective pipeline:

1. Scrapers fetch job postings
2. Data is cleaned and enriched with normalized fields
3. Jobs are written to a canonical JSON dataset
4. The website reads from that dataset through a cached reader
5. The search index is built from the same jobs
6. The agent uses semantic search and job metadata to recommend opportunities
7. The browser UI renders the job board and the voice assistant can open jobs on the page

This keeps the browser UI and the search index aligned by using shared job IDs and a single data source.

## Repository layout

- `agent.py` — terminal-friendly LangGraph agent for testing the conversational logic
- `ui_tools.py` — the tools the agent uses to update the web page and search jobs
- `job_tools.py` — query parsing for pay/recency filters and structured job lookup
- `vector_store.py` — Qdrant search index, embeddings, reranking, and job retrieval
- `jobs_data.py` — canonical reader for the job dataset used by the web app
- `classify.py` — job enrichment, normalization, categorization, and pay formatting
- `ingest.py` — ingestion pipeline for saving jobs and syncing them with the search index
- `web_scraper.py` — site scraping and HTML/markdown cleanup
- `mercor.py` — additional job page parsing and markdown cleanup logic
- `server.py` — FastAPI app serving the board, API, and LiveKit token generation
- `voice_agent.py` — LiveKit-based voice assistant and conversation pipeline
- `public/` — static frontend assets and the board page
- `langgraph.json` — LangGraph configuration for the agent
- `Dockerfile` — worker container for the voice assistant
- `requirements.txt` — core Python dependencies
- `requirements-local.txt` — local dev dependency bundle
- `.gitignore`, `.dockerignore` — repo/package hygiene files

## Core modules

### `agent.py`

This is the main LangGraph agent used by the application logic. It builds an agent graph using a DeepSeek model and connects it to the UI tools.

Responsibilities:
- define the system prompt for the job assistant
- stream only the spoken output to the voice layer
- create the agent graph for terminal testing or Studio use
- allow the same agent logic to be reused by the voice pipeline

The module is designed so the conversation is natural and brief in voice mode: short answers, no markdown, no URLs, no long lists.

### `ui_tools.py`

This is the job agent’s action layer. It exposes the actual tools the model can call to:
- search jobs
- open a job on the page
- highlight the Apply button
- restore the full list of jobs

Important design points:
- every tool sends page updates through a callback
- search results are powered by the vector database
- the displayed board always reads jobs from the same JSON used by the website
- tool output is kept concise and structured for the LLM

The tool layer is intentionally separated from the frontend so the page and agent can evolve independently while staying synchronized.

### `job_tools.py`

This file is the natural-language query engine for job searches.

It:
- parses pay constraints such as “over $50/hour” or “under $100/hour”
- parses recency like “this week” or “last month”
- strips the filter language from the semantic search string
- passes the remaining wording to the vector DB
- formats results for agent consumption

This is critical because semantic embedding search is strong at intent matching but weaker at precise numeric/date enforcement.

### `vector_store.py`

This is the vector search system that powers semantic job retrieval.

It includes:
- Qdrant client configuration
- embedding model setup
- reranker configuration
- collection schema for job metadata
- semantic search, filtering, and retrieval
- normalization of job payloads
- pay and post-date extraction helpers

The data model is built around a compact “search card” containing the job’s essential searchable information, while the full description stays in the canonical dataset. This keeps embeddings smaller and faster while preserving the richer dataset for display.

### `jobs_data.py`

This module is the canonical reader for the project’s job dataset. It is designed to guarantee that the API and the browser UI are reading from the same JSON file.

Responsibilities:
- cache the JSON file contents
- reload only when the file changes
- enrich each job upon read
- build searchable facets like company, category, skill, and level
- expose single-job and multi-job fetching
- support list filtering and sorting

This module is deliberately the single path through which the website reads jobs. It prevents the UI from drifting away from the search index or stale scraped results.

### `classify.py`

This is the normalization and enrichment layer for each job.

It likely handles:
- category attribution
- domain and role classification
- pay label generation
- level and employment extraction
- skill normalization
- text cleanup and enrichment

It ensures scraped job data becomes consistent enough to query, filter, and render with confidence.

### `ingest.py`

This script turns scraped raw data into a usable dataset and search index.

It is responsible for:
- loading scraped jobs
- cleaning and validating entries
- writing the canonical JSON dataset
- inferring sources and metadata
- making sure the vector store is updated consistently with the dataset

The repo is designed so re-ingesting the same source data re-syncs the board and the search layer.

### `web_scraper.py`

This is the scraping engine for collecting jobs from public sources.

It manages:
- crawling pages
- handling URLs and sanitization
- cleaning markdown or content strings
- extracting the meaningful job information from messy HTML
- harvesting the job listing content that later gets normalized and indexed

This is the ingestion boundary where raw website output becomes structured project data.

### `mercor.py`

This module adds another parsing layer to convert scraped website content into a cleaner, more uniform format.

It is focused on:
- stripping noise from job description text
- cleaning section headings
- reducing markdown clutter
- preserving the core content needed for search and display

This is especially useful when job sites inject a lot of layout noise, duplicated sections, or inconsistent formatting.

### `server.py`

This file runs the FastAPI website and API.

It exposes:
- `GET /` — serves the board page
- `GET /api/jobs` — paginated and filtered job list
- `GET /api/jobs/{job_id}` — one full job
- `GET /api/health` — deployment health check
- `GET /api/version` — version string for refresh logic
- `GET /token` — LiveKit room token and dispatch configuration

The app is designed to work:
- locally with a Python server
- in a deployment where static files are served via CDN
- with LiveKit for voice room access

This file is also where the app confirms that required environment variables are present.

### `voice_agent.py`

This is the voice assistant runtime.

It integrates:
- LiveKit room connections
- speech-to-text and text-to-speech
- the job agent graph
- a page-side UI flow for live job recommendations

The assistant opens job cards, follows user intent, and speaks short natural responses. The page can also show status messages and highlight the Apply button when the user wants to proceed.

### `public/`

The frontend is a browser-only job board.

- `public/index.html` — main page shell
- `public/static/site.css` — board and UI styling
- `public/static/site.js` — the page logic that loads from `/api/jobs`
- `public/static/orb.js` — animated orb/assistant visual effects
- `public/static/voice.js` — LiveKit voice front-end behavior

The page is intentionally decoupled from messaging originating in the Python agent. Instead, the agent sends job IDs, and the front end fetches the data it needs from the API. This reduces the chance of front-end drift or stale data.

## Data flow

The app is built around a predictable sequence:

1. Scraped jobs are captured
2. They are normalized and enriched
3. They are saved to the canonical dataset
4. The website reads from that dataset via `jobs_data.py`
5. Embeddings are created for semantic search in `vector_store.py`
6. The agent searches through the vector store and retrieves matching job IDs
7. The UI opens the chosen jobs using those IDs
8. The user speaks or clicks and the board updates accordingly

This makes the system consistent even as the data evolves.

## Runtime and setup

### Local development

Install the base dependencies:

```bash
pip install -r requirements.txt
