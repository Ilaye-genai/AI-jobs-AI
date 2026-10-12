"use strict";
/*
 * The job board. It knows nothing about LiveKit or the voice agent: the agent just calls one
 * method on window.SITE (bottom of this file) and the page does the rest. With the agent off,
 * the board is a normal site.
 *
 * Data comes only from /api/jobs, which reads scraped_jobs.json. Nothing here depends on which
 * jobs, categories or skills exist, so a new scrape shows up with no frontend change.
 *
 * Open the site with ?dev=1 (http://localhost:8080/?dev=1) for a panel that plays the agent's part
 * with buttons, so you can test the page movement without voice and without the browser console.
 */
(() => {
  const $ = (sel, root = document) => root.querySelector(sel);
  const app = $("#app");
  const notice = $("#notice");
  const searchBox = $("#q");

  const esc = (s) =>
    String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  // Scraped data is untrusted: only ever link to http(s) URLs (never javascript:).
  const safeUrl = (u) => (/^https?:\/\//i.test(u || "") ? u : "#");

  const state = {
    filters: { q: "", category: "", level: "", employment: "", min_pay: "", sort: "new" },
    picks: null, // { jobs } while the assistant has chosen what the board shows
    facets: null,
    version: null,
    loadToken: 0, // lets a newer request cancel the display of an older one
    pendingHighlight: null,
  };

  // ------------------------------------------------------------------ helpers
  async function api(path, params) {
    const qs = new URLSearchParams();
    for (const [k, v] of Object.entries(params || {})) if (v !== "" && v != null) qs.set(k, v);
    const query = qs.toString();
    const res = await fetch(path + (query ? "?" + query : ""));
    if (!res.ok) throw new Error(res.status === 404 ? "Not found" : "Server error " + res.status);
    return res.json();
  }

  function ago(iso) {
    if (!iso) return "";
    const days = (Date.now() - new Date(iso).getTime()) / 864e5;
    if (days < 1 / 24) return "just now";
    if (days < 1) return Math.floor(days * 24) + "h ago";
    if (days < 2) return "yesterday";
    if (days < 30) return Math.floor(days) + " days ago";
    return new Date(iso).toLocaleDateString();
  }

  const join = (parts) => parts.filter(Boolean).map(esc).join(" · ");

  // Minimal, safe markdown for job descriptions: escape first, then add structure.
  function renderBody(md) {
    const inline = (t) => esc(t).replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
    let html = "";
    let list = null;
    const close = () => {
      if (list) html += `</${list}>`;
      list = null;
    };
    const item = (kind, text) => {
      if (list !== kind) {
        close();
        html += `<${kind}>`;
        list = kind;
      }
      html += `<li>${inline(text)}</li>`;
    };
    for (const raw of String(md || "").split("\n")) {
      const line = raw.trim();
      let m;
      if (!line) close();
      else if ((m = line.match(/^#{1,6}\s+(.*)/))) {
        close();
        html += `<h3>${inline(m[1])}</h3>`;
      } else if ((m = line.match(/^\d+[.)]\s+(.*)/))) item("ol", m[1]);
      else if ((m = line.match(/^[-*•]\s+(.*)/))) item("ul", m[1]);
      else if (/^\*\*[^*]+\*\*:?$/.test(line)) {
        close();
        html += `<h3>${esc(line.replace(/\*\*/g, "").replace(/:$/, ""))}</h3>`; // a bold-only line is a section title
      } else {
        close();
        html += `<p>${inline(line)}</p>`;
      }
    }
    close();
    return html;
  }

  // ------------------------------------------------------------------ list view
  function cardHtml(job, i, isPick) {
    const chips = job.skills
      .filter((s) => s.length <= 28)
      .slice(0, 4)
      .map((s) => `<li>${esc(s)}</li>`)
      .join("");
    return `
      <a class="card${isPick ? " pick" : ""}" style="--i:${i}" href="#/job/${encodeURIComponent(job.id)}" data-id="${esc(job.id)}">
        <div class="card-top"><span class="tag">${esc(job.category)}</span><span>${esc(ago(job.posted))}</span></div>
        <h2>${esc(job.title)}</h2>
        <p class="sub">${join([job.company, job.employment, job.location])}</p>
        ${job.pay_label ? `<p class="pay">${esc(job.pay_label)}${job.pay_model === "per-task" ? '<span class="note">paid per task</span>' : ""}</p>` : ""}
        ${job.summary ? `<p class="sum">${esc(job.summary)}</p>` : ""}
        ${chips ? `<ul class="chips">${chips}</ul>` : ""}
      </a>`;
  }

  function renderPicksBanner() {
    const el = $("#picks");
    const hero = $("#hero");
    if (hero) hero.hidden = !!state.picks; // the assistant's picks replace the invitation to talk
    if (!el) return;
    if (!state.picks) {
      el.innerHTML = "";
      return;
    }
    const n = state.picks.jobs.length;
    el.innerHTML = `
      <div class="picks-banner">
        <strong>Picked for you by the assistant</strong>
        <span class="muted">${n} job${n === 1 ? "" : "s"}, best match first</span>
        <button class="btn ghost" type="button" id="show-all">Show all jobs</button>
      </div>`;
    $("#show-all").addEventListener("click", showAll);
  }

  function renderFilters() {
    const box = $("#filters");
    if (!box || !state.facets) return;
    const f = state.filters;
    const select = (key, label, facet) => {
      const options = (state.facets[facet] || []).map(
        (o) => `<option value="${esc(o.value)}"${f[key] === o.value ? " selected" : ""}>${esc(o.value)} (${o.count})</option>`
      );
      if (!options.length) return "";
      return `<select data-key="${key}" aria-label="${label}"><option value="">${label}</option>${options.join("")}</select>`;
    };
    const pay = [30, 50, 75, 100]
      .map((n) => `<option value="${n}"${String(f.min_pay) === String(n) ? " selected" : ""}>$${n}+ /hr</option>`)
      .join("");
    box.innerHTML =
      select("category", "All fields", "category") +
      select("level", "Any level", "level") +
      select("employment", "Any type", "employment") +
      `<select data-key="min_pay" aria-label="Minimum pay"><option value="">Any pay</option>${pay}</select>` +
      `<select data-key="sort" aria-label="Sort"><option value="new"${f.sort === "new" ? " selected" : ""}>Newest first</option>` +
      `<option value="pay"${f.sort === "pay" ? " selected" : ""}>Highest pay</option></select>`;
  }

  function paint(jobs, total, isPick, append = false) {
    const grid = $("#grid");
    const meta = $("#meta");
    if (!grid) return;
    if (!jobs.length && !append) {
      grid.innerHTML = `<div class="empty" style="grid-column:1/-1"><p>No jobs match those filters.</p><button class="btn" type="button" id="clear">Clear filters</button></div>`;
      meta.textContent = "";
      $("#clear").addEventListener("click", () => {
        state.filters = { q: "", category: "", level: "", employment: "", min_pay: "", sort: "new" };
        searchBox.value = "";
        loadList();
        renderFilters();
      });
      return;
    }
    const html = jobs.map((j, i) => cardHtml(j, i, isPick)).join("");
    if (append) grid.insertAdjacentHTML("beforeend", html);
    else grid.innerHTML = html;
    const shown = grid.querySelectorAll(".card").length;
    meta.innerHTML = `<span>${total} job${total === 1 ? "" : "s"}${shown < total ? `, showing ${shown}` : ""}</span>`;
    if (!isPick && shown < total) {
      meta.innerHTML += `<button class="linklike" type="button" id="more">Show more</button>`;
      $("#more").addEventListener("click", () => loadList(true));
    }
  }

  async function loadList(more = false) {
    const grid = $("#grid");
    if (!grid) return;
    const token = ++state.loadToken;
    renderPicksBanner();
    if (state.picks) {
      paint(state.picks.jobs, state.picks.jobs.length, true);
      return;
    }
    if (!more) grid.innerHTML = '<p class="empty" style="grid-column:1/-1">Loading jobs…</p>';
    try {
      const offset = more ? grid.querySelectorAll(".card").length : 0;
      const data = await api("/api/jobs", { ...state.filters, offset });
      if (token !== state.loadToken) return; // a newer request replaced this one
      state.version = data.version;
      if (data.facets && JSON.stringify(data.facets) !== JSON.stringify(state.facets)) {
        state.facets = data.facets;
        renderFilters();
      }
      paint(data.jobs, data.total, false, more);
    } catch (e) {
      grid.innerHTML = `<p class="empty" style="grid-column:1/-1">Could not load jobs: ${esc(e.message)}</p>`;
    }
  }

  function showList() {
    document.title = "AI Jobs";
    document.body.classList.remove("split-view");
    app.innerHTML = `
      <section class="hero" id="hero">
        <div>
          <h1>Tell the assistant what you want. It finds the job and opens it for you.</h1>
          <p>Say your field, pay and hours out loud. It searches the board, recommends the best match and takes you straight to it.</p>
        </div>
        <button class="btn big" type="button" id="hero-start">Talk to the assistant</button>
      </section>
      <div id="picks"></div><div id="filters" class="filters"></div><div id="meta" class="meta"></div><div id="grid" class="grid"></div>`;
    $("#hero-start").addEventListener("click", () => window.VOICE?.toggle());
    renderFilters();
    $("#filters").addEventListener("change", (e) => {
      const key = e.target.dataset.key;
      if (!key) return;
      state.filters[key] = e.target.value;
      state.picks = null; // choosing a filter means "stop showing the assistant's picks"
      loadList();
    });
    window.scrollTo({ top: 0 });
    loadList();
  }

  function showAll() {
    state.picks = null;
    go("#/");
  }

  // ------------------------------------------------------------------ detail view
  function detailHtml(job) {
    const skills = job.skills.map((s) => `<li>${esc(s)}</li>`).join("");
    const original = esc(safeUrl(job.url));
    const body = job.has_description
      ? renderBody(job.body)
      : `<p class="nodesc">The scraper only captured the top of this posting, not its description, so there is nothing to show here yet.
           <a href="${original}" target="_blank" rel="noopener noreferrer">Read the full posting on ${esc(job.company)} ↗</a></p>`;
    return `
      <article class="detail">
        <a class="back" href="#/">← ${state.picks ? "Back to the assistant's picks" : "All jobs"}</a>
        <div><span class="tag">${esc(job.category)}</span></div>
        <h1>${esc(job.title)}</h1>
        <p class="sub">${join([job.company, job.employment, job.location, job.hours, job.level, job.posted ? "Posted " + ago(job.posted) : ""])}</p>
        ${job.pay_label ? `<p class="pay big">${esc(job.pay_label)}</p>` : ""}
        ${job.pay_model === "per-task" ? '<p class="muted">The posting says pay is output-based (paid per task), so the hourly range is approximate.</p>' : ""}
        ${skills ? `<ul class="chips">${skills}</ul>` : ""}
        <div class="actions">
          <a id="apply" class="btn big" href="${esc(safeUrl(job.apply_url || job.url))}" target="_blank" rel="noopener noreferrer">Apply on ${esc(job.company)} ↗</a>
          ${job.has_description ? `<a class="btn outline" href="${original}" target="_blank" rel="noopener noreferrer">Read the original posting ↗</a>` : ""}
        </div>
        <div class="body">${body}</div>
      </article>`;
  }

  function pulseApply() {
    const b = $("#apply");
    if (!b) return;
    b.scrollIntoView({ block: "center", behavior: "smooth" });
    b.classList.remove("pulse");
    void b.offsetWidth; // restart the animation if it already ran
    b.classList.add("pulse");
  }

  // ------------------------------------------------------------------ split view
  // While the assistant has picks and a job is open, the page is split: the picks stay in a narrow list
  // on the left, the opened job (description first) fills the big pane on the right.
  function miniHtml(job, selected) {
    return `
      <a class="mini${selected ? " selected" : ""}" href="#/job/${encodeURIComponent(job.id)}" data-id="${esc(job.id)}"${selected ? ' aria-current="true"' : ""}>
        <span class="tag">${esc(job.category)}</span>
        <strong class="mini-title">${esc(job.title)}</strong>
        <span class="sub">${join([job.company, job.employment, job.location])}</span>
        ${job.pay_label ? `<span class="pay">${esc(job.pay_label)}${job.pay_model === "per-task" ? '<span class="note">paid per task</span>' : ""}</span>` : ""}
      </a>`;
  }

  function splitListHtml(selectedId) {
    const jobs = state.picks.jobs;
    return `
      <div class="split-head">
        <strong>Picked for you</strong>
        <span class="muted">${jobs.length} job${jobs.length === 1 ? "" : "s"}</span>
        <button class="linklike" type="button" id="split-all">Show all jobs</button>
      </div>
      <div class="split-items">${jobs.map((j) => miniHtml(j, j.id === selectedId)).join("")}</div>`;
  }

  function mountSplit(selectedId) {
    app.innerHTML = `<div class="split" id="split">
      <aside class="split-list" id="split-list" aria-label="Other recommendations">${splitListHtml(selectedId)}</aside>
      <section class="split-detail" id="split-detail" aria-live="polite"></section>
    </div>`;
    $("#split-all").addEventListener("click", showAll);
  }

  // Highlight the open job in the left list without rebuilding it, so its scroll position stays put.
  function selectInSplit(id) {
    const box = $("#split-list .split-items");
    document.querySelectorAll("#split-list .mini").forEach((el) => {
      const on = el.dataset.id === id;
      el.classList.toggle("selected", on);
      if (on) el.setAttribute("aria-current", "true");
      else el.removeAttribute("aria-current");
      if (on && box) {
        const r = el.getBoundingClientRect();
        const b = box.getBoundingClientRect();
        if (r.top < b.top) box.scrollTop -= b.top - r.top;
        else if (r.bottom > b.bottom) box.scrollTop += r.bottom - b.bottom;
      }
    });
  }

  async function showJob(id) {
    const token = ++state.loadToken;
    const split = !!(state.picks && state.picks.jobs.length);
    document.body.classList.toggle("split-view", split);
    let target = app;
    if (split) {
      if (!$("#split")) mountSplit(id);
      selectInSplit(id);
      target = $("#split-detail");
    }
    target.innerHTML = '<p class="empty">Loading…</p>';
    try {
      const job = await api("/api/jobs/" + encodeURIComponent(id));
      if (token !== state.loadToken) return;
      target.innerHTML = detailHtml(job);
      document.title = `${job.title} · AI Jobs`;
      window.scrollTo({ top: 0 });
      if (state.pendingHighlight === id) {
        state.pendingHighlight = null;
        setTimeout(pulseApply, 350);
      }
    } catch (e) {
      if (token !== state.loadToken) return;
      const msg = e.message === "Not found" ? "That job is no longer listed." : esc(e.message);
      target.innerHTML = `<div class="detail"><a class="back" href="#/">← ${split ? "All picks" : "All jobs"}</a><p class="empty">${msg}</p></div>`;
    }
  }

  // ------------------------------------------------------------------ routing
  function route() {
    const m = location.hash.match(/^#\/job\/([^/?#]+)/);
    if (m) showJob(decodeURIComponent(m[1]));
    else showList();
  }

  // Changing location.hash to the value it already has fires no event, so re-run the route ourselves.
  function go(hash) {
    if (location.hash === hash) route();
    else location.hash = hash;
  }

  window.addEventListener("hashchange", route);

  let searchTimer;
  searchBox.addEventListener("input", () => {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(() => {
      state.filters.q = searchBox.value.trim();
      state.picks = null;
      if ($("#grid")) loadList();
      else go("#/");
    }, 250);
  });

  // ------------------------------------------------------------------ new-scrape notice
  function showNotice() {
    notice.hidden = false;
    notice.innerHTML = '<span>New jobs were added.</span><button type="button" id="refresh">Refresh</button>';
    $("#refresh").addEventListener("click", async () => {
      notice.hidden = true;
      try {
        state.version = (await api("/api/version")).version;
      } catch {}
      route();
    });
  }

  setInterval(async () => {
    try {
      const { version } = await api("/api/version");
      if (state.version && version !== state.version && notice.hidden) showNotice();
    } catch {}
  }, 30000);

  // ------------------------------------------------------------------ what the voice agent can do
  // The agent only ever sends Job IDs; the page loads the jobs itself from /api/jobs.
  async function show_results({ ids }) {
    const list = (ids || []).filter(Boolean);
    if (!list.length) return;
    try {
      const data = await api("/api/jobs", { ids: list.join(",") });
      if (!data.jobs.length) return;
      state.picks = { jobs: data.jobs };
      if (!state.version) state.version = data.version;
      if ($("#split")) {
        // A job is already open next to the picks: refresh the list, keep the job on screen.
        const m = location.hash.match(/^#\/job\/([^/?#]+)/);
        $("#split-list").innerHTML = splitListHtml(m ? decodeURIComponent(m[1]) : null);
        $("#split-all").addEventListener("click", showAll);
      } else go("#/");
    } catch (e) {
      console.error("show_results failed", e);
    }
  }

  function show_job({ job_id }) {
    if (job_id) go("#/job/" + encodeURIComponent(job_id));
  }

  function highlight_apply({ job_id }) {
    if (!job_id) return;
    state.pendingHighlight = job_id;
    go("#/job/" + encodeURIComponent(job_id));
  }

  function show_all() {
    showAll();
  }

  // The agent says what it is doing ("Searching jobs…"). The orb shows it big, under the animation.
  // An empty text means the action is over.
  function status({ text, kind, sub }) {
    if (window.ORB) {
      if (text) window.ORB.action(String(text).slice(0, 80), kind, sub);
      else window.ORB.endAction();
      return;
    }
    const label = $("#voice-label"); // fallback when orb.js is missing
    if (label && text) label.textContent = String(text).slice(0, 80);
  }

  window.SITE = { show_results, show_job, highlight_apply, show_all, status };

  // ------------------------------------------------------------------ test panel (?dev=1)
  // Plays the agent's part with buttons, so the page can be tested with no voice and no console.
  function mountDevPanel() {
    const panel = document.createElement("div");
    panel.className = "devpanel";
    panel.innerHTML = `
      <strong>Test the page (no voice)</strong>
      <button type="button" data-t="picks">1. Show 3 picks</button>
      <button type="button" data-t="job">2. Open the first pick</button>
      <button type="button" data-t="apply">3. Pulse the Apply button</button>
      <button type="button" data-t="all">4. Back to all jobs</button>`;
    document.body.appendChild(panel);
    let ids = [];
    panel.addEventListener("click", async (e) => {
      const t = e.target.dataset.t;
      if (!t) return;
      if (!ids.length) {
        try {
          ids = (await api("/api/jobs", { limit: 3 })).jobs.map((j) => j.id);
        } catch (err) {
          console.error("dev panel: could not load jobs", err);
          return;
        }
      }
      if (!ids.length) return;
      if (t === "picks") window.SITE.show_results({ ids });
      if (t === "job") window.SITE.show_job({ job_id: ids[0] });
      if (t === "apply") window.SITE.highlight_apply({ job_id: ids[0] });
      if (t === "all") window.SITE.show_all();
    });
  }
  if (new URLSearchParams(location.search).has("dev")) mountDevPanel();

  // ------------------------------------------------------------------ start
  api("/api/version").then((v) => (state.version = state.version || v.version)).catch(() => {});
  route();
})();