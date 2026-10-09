"""
Resume Review Agent
-------------------
A beginner-friendly app built with:
  - Streamlit  -> the user interface
  - CrewAI     -> 1 Agent + 1 Task + 1 Crew
  - Groq       -> the LLM (default: openai/gpt-oss-120b)
  - pypdf      -> reads text from PDF resumes

Secrets (Streamlit Cloud -> App settings -> Secrets):
  GROQ_API_KEY = "gsk_..."
  GROQ_MODEL   = "openai/gpt-oss-120b"   # optional
"""

import html
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

# Some hosting machines have an old SQLite that CrewAI dislikes.
# If pysqlite3 is installed, we use it instead. If not, we simply continue.
try:
    import pysqlite3  # noqa: F401

    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except Exception:
    pass

# Turn off anonymous telemetry (keeps the app quiet and fast).
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")

import streamlit as st
from pypdf import PdfReader

# ----------------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Resume Review Agent",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_MODEL = "openai/gpt-oss-120b"
MAX_RESUME_CHARS = 12000  # keeps us inside Groq token limits
MAX_JD_CHARS = 6000
UNKNOWN = "Unknown / Not Demonstrated"

SAMPLE_RESUME = """Mahnoor
Karachi, Pakistan | Mahnoor.khan@example.com

SUMMARY
Marketing graduate with 2 years of experience in social media and content creation.

EXPERIENCE
Social Media Executive - BrightWave Agency (2023 - Present)
- Managed Instagram and Facebook pages for 5 clients
- Created weekly content calendars and posted daily
- Worked with the design team on campaign visuals

Content Intern - Sprout Media (2022 - 2023)
- Wrote blog posts and short captions
- Helped with basic reporting in Google Sheets

EDUCATION
BBA Marketing - University of Karachi (2022)

SKILLS
Canva, Google Sheets, Meta Business Suite, Copywriting
"""

SAMPLE_JD = """Digital Marketing Specialist

We are looking for a Digital Marketing Specialist to plan and run online campaigns.

Requirements:
- 2+ years of experience in digital marketing
- Experience running paid ads on Meta or Google Ads
- Strong copywriting skills
- Ability to analyze campaign performance using Google Analytics
- Experience with email marketing tools (e.g., Mailchimp)
- Good communication and teamwork
"""

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
st.markdown(
    """
<style>
.block-container {padding-top: 1.6rem; max-width: 1200px;}
.hero {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 55%, #db2777 100%);
    padding: 2rem 2.2rem; border-radius: 20px; color: white; margin-bottom: 1.2rem;
    box-shadow: 0 10px 30px rgba(79,70,229,.25);
}
.hero h1 {margin: 0; font-size: 2.2rem; color: white;}
.hero p {margin: .4rem 0 0 0; font-size: 1.05rem; opacity: .95;}
.step-card {
    border: 1px solid rgba(128,128,128,.25); border-radius: 14px;
    padding: .9rem 1.1rem; margin-bottom: .6rem;
}
.score-wrap {display:flex; align-items:center; gap:1.4rem; padding: 1rem 1.2rem;
    border:1px solid rgba(128,128,128,.25); border-radius:18px; margin-bottom:1rem;}
.score-ring {width:120px; height:120px; border-radius:50%; display:flex;
    align-items:center; justify-content:center; flex-shrink:0;}
.score-inner {width:92px; height:92px; border-radius:50%; background:var(--background-color, #fff);
    display:flex; flex-direction:column; align-items:center; justify-content:center;}
.score-inner b {font-size:1.9rem; line-height:1;}
.score-inner span {font-size:.7rem; opacity:.7;}
.pill {display:inline-block; padding:.15rem .7rem; border-radius:999px; font-size:.78rem;
    font-weight:600; margin-right:.4rem;}
.pill-green {background:#dcfce7; color:#166534;}
.pill-yellow {background:#fef9c3; color:#854d0e;}
.pill-gray {background:#e5e7eb; color:#374151;}
.pill-blue {background:#dbeafe; color:#1e40af;}
.item-card {border:1px solid rgba(128,128,128,.25); border-radius:12px;
    padding:.8rem 1rem; margin-bottom:.6rem;}
.item-card small {opacity:.75;}
.tag {display:inline-block; background:rgba(124,58,237,.12); color:#7c3aed;
    padding:.2rem .65rem; border-radius:8px; margin:.15rem .25rem .15rem 0; font-size:.85rem;}
.footer-note {text-align:center; opacity:.6; font-size:.85rem; margin-top:2rem;}
</style>
""",
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------
def esc(value) -> str:
    """Make text safe to show inside HTML."""
    return html.escape(str(value if value is not None else ""))


def get_secrets():
    """Read the API key and model from Streamlit secrets. Returns (key, model)."""
    try:
        key = st.secrets["GROQ_API_KEY"]
    except Exception:
        return None, DEFAULT_MODEL
    try:
        model = st.secrets.get("GROQ_MODEL") or DEFAULT_MODEL
    except Exception:
        model = DEFAULT_MODEL
    return (key or "").strip() or None, model.strip()


def extract_pdf_text(uploaded_file):
    """Read text from a PDF. Returns (text, error_message)."""
    try:
        reader = PdfReader(uploaded_file)
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception:
                return "", "This PDF is password-protected. Please upload an unlocked copy."
        pages = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        text = "\n".join(pages).strip()
    except Exception:
        return "", (
            "We could not read this PDF. The file may be corrupted. "
            "Try re-saving it as a PDF, or paste your resume text instead."
        )
    if len(text) < 30:
        return "", (
            "No readable text was found. This often means the PDF is a scanned image. "
            "Please paste your resume text instead, or upload a text-based PDF."
        )
    return text, None


def friendly_error(exc: Exception) -> str:
    """Turn technical errors into simple messages for students."""
    msg = str(exc).lower()
    if isinstance(exc, ImportError) or "litellm" in msg:
        return (
            "📦 A required library is missing (litellm). Run `pip install -r requirements.txt` "
            "again (or redeploy on Streamlit Cloud), then try again."
        )
    if "rate limit" in msg or "429" in msg or "rate_limit" in msg or "tokens per" in msg:
        return (
            "⏳ Groq is receiving too many requests right now (rate limit). "
            "Please wait about a minute and try again. Shorter resume/job text also helps."
        )
    if "timeout" in msg or "timed out" in msg:
        return "⌛ The request took too long (timeout). Please try again in a moment."
    if "401" in msg or "invalid api key" in msg or "authentication" in msg or "unauthorized" in msg:
        return "🔑 Your Groq API key looks invalid. Please check GROQ_API_KEY in your Streamlit secrets."
    if "decommission" in msg or "model_not_found" in msg or "does not exist" in msg or "404" in msg:
        return (
            "🤖 The selected Groq model is not available. Set GROQ_MODEL in secrets to an active "
            "model such as openai/gpt-oss-120b."
        )
    if "connection" in msg or "network" in msg:
        return "🌐 Could not connect to Groq. Please check your internet connection and try again."
    return "⚠️ Something went wrong while reviewing your resume. Please try again in a moment."


def test_groq_connection(api_key: str):
    """Quick check: can this app reach Groq and is the key accepted? Returns (ok, message)."""
    req = urllib.request.Request(
        "https://api.groq.com/openai/v1/models",
        headers={"Authorization": f"Bearer {api_key}", "User-Agent": "resume-review-agent"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        ids = [m.get("id") for m in body.get("data", [])]
        return True, ids
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            return False, f"Groq was reached, but the API key was rejected (HTTP {e.code}). Create a new key."
        return False, f"Groq answered with HTTP {e.code}."
    except Exception as e:  # noqa: BLE001
        return False, (
            "This computer/server cannot reach api.groq.com "
            f"({type(e).__name__}). Check internet, VPN, firewall or ISP blocking."
        )


def redact(text: str, secret) -> str:
    """Hide the API key if it ever appears in an error message."""
    text = str(text)
    if secret:
        text = text.replace(secret, "***")
    return re.sub(r"gsk_[A-Za-z0-9]+", "gsk_***", text)[:800]


def build_task_description(resume, jd, tone, focus):
    return f"""
You are reviewing a candidate's resume against a target job description.

TONE: {tone}
EXTRA FOCUS AREAS: {", ".join(focus) if focus else "None"}

STRICT RULES:
1. Use ONLY facts that appear in the resume text. NEVER invent skills, tools, degrees,
   employers, numbers, or achievements.
2. If the resume does not clearly show something the job requires, mark it exactly as
   "{UNKNOWN}". Do not guess.
3. Rewrite suggestions may only rephrase existing resume content. If a number or detail
   is missing, use a placeholder such as [add number] and ask the candidate to fill it in
   ONLY if it is true.
4. Be specific, practical and encouraging. Keep every item short.

=== RESUME START ===
{resume}
=== RESUME END ===

=== JOB DESCRIPTION START ===
{jd}
=== JOB DESCRIPTION END ===

Return ONLY one valid JSON object (no markdown, no extra text) with exactly these keys:
- "match_score": integer 0-100 (how well the resume currently matches the job)
- "verdict": short phrase, e.g. "Good match with some gaps"
- "summary": 2-3 sentence overview
- "strengths": list of short strings
- "requirements": list of objects with keys "requirement" (from the job description),
  "status" (one of "Demonstrated", "Partially Demonstrated", "{UNKNOWN}"),
  "evidence" (what in the resume supports it, or "No evidence found in resume")
- "gaps": list of short strings (missing or weak areas)
- "missing_keywords": list of important job keywords NOT in the resume
  (only suggest adding them if the candidate truly has that experience)
- "rewrite_suggestions": list of objects with keys "section", "current",
  "suggested", "reason"
- "formatting_tips": list of short strings about clarity / ATS friendliness
- "next_steps": list of 3-5 prioritized action items
""".strip()


def parse_json_output(raw: str):
    """Try to turn the model's answer into a Python dict. Returns dict or None."""
    if not raw:
        return None
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        data = json.loads(text[start : end + 1])
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def run_review(resume, jd, api_key, model, tone, focus):
    """The heart of the app: 1 Agent + 1 Task + 1 Crew."""
    from crewai import LLM, Agent, Crew, Process, Task

    model_name = model if model.startswith("groq/") else f"groq/{model}"
    llm = LLM(model=model_name, api_key=api_key, temperature=0.2, timeout=120)

    agent = Agent(
        role="Senior Resume Reviewer",
        goal="Honestly evaluate how well a resume matches a job description and give "
        "clear, truthful, actionable improvements.",
        backstory="You are an experienced recruiter and career coach. You never invent "
        "qualifications. When evidence is missing you say so plainly.",
        llm=llm,
        allow_delegation=False,
        verbose=False,
    )
    task = Task(
        description=build_task_description(resume, jd, tone, focus),
        expected_output="A single valid JSON object following the requested keys.",
        agent=agent,
    )
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)

    last_error = None
    for attempt in range(2):  # one automatic retry for rate limits
        try:
            result = crew.kickoff()
            return getattr(result, "raw", None) or str(result)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            m = str(exc).lower()
            if attempt == 0 and ("rate limit" in m or "429" in m):
                time.sleep(8)
                continue
            break
    raise last_error


def score_style(score: int):
    if score >= 75:
        return "#16a34a"
    if score >= 50:
        return "#f59e0b"
    return "#ef4444"


def status_pill(status: str) -> str:
    s = (status or "").lower()
    if s.startswith("demonstrated"):
        return '<span class="pill pill-green">✅ Demonstrated</span>'
    if s.startswith("partially"):
        return '<span class="pill pill-yellow">🟡 Partially Demonstrated</span>'
    return f'<span class="pill pill-gray">❔ {esc(UNKNOWN)}</span>'


def build_report_markdown(d: dict) -> str:
    lines = [
        "# Resume Review Report",
        f"**Match score:** {d.get('match_score', 0)}/100 - {d.get('verdict', '')}",
        "",
        "## Summary",
        d.get("summary", ""),
        "",
        "## Strengths",
        *[f"- {x}" for x in d.get("strengths", [])],
        "",
        "## Requirements check",
    ]
    for r in d.get("requirements", []):
        lines.append(f"- **{r.get('requirement')}** - {r.get('status')} ({r.get('evidence')})")
    lines += ["", "## Gaps", *[f"- {x}" for x in d.get("gaps", [])]]
    lines += ["", "## Keywords to consider (only if true for you)"]
    lines += [f"- {x}" for x in d.get("missing_keywords", [])]
    lines += ["", "## Rewrite suggestions"]
    for r in d.get("rewrite_suggestions", []):
        lines += [
            f"### {r.get('section')}",
            f"- Current: {r.get('current')}",
            f"- Suggested: {r.get('suggested')}",
            f"- Why: {r.get('reason')}",
        ]
    lines += ["", "## Formatting tips", *[f"- {x}" for x in d.get("formatting_tips", [])]]
    lines += ["", "## Next steps", *[f"{i}. {x}" for i, x in enumerate(d.get("next_steps", []), 1)]]
    return "\n".join(lines)


def show_results(d: dict):
    try:
        score = max(0, min(100, int(d.get("match_score", 0))))
    except Exception:
        score = 0
    color = score_style(score)

    st.markdown(
        f"""
<div class="score-wrap">
  <div class="score-ring" style="background: conic-gradient({color} {score * 3.6}deg, rgba(128,128,128,.2) 0deg);">
    <div class="score-inner"><b>{score}</b><span>out of 100</span></div>
  </div>
  <div>
    <h3 style="margin:0">{esc(d.get("verdict", "Review complete"))}</h3>
    <p style="margin:.3rem 0 0 0; opacity:.85">{esc(d.get("summary", ""))}</p>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    reqs = d.get("requirements", []) or []
    demo = sum(1 for r in reqs if str(r.get("status", "")).lower().startswith("demonstrated"))
    part = sum(1 for r in reqs if str(r.get("status", "")).lower().startswith("partially"))
    unk = len(reqs) - demo - part
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Requirements checked", len(reqs))
    c2.metric("✅ Demonstrated", demo)
    c3.metric("🟡 Partial", part)
    c4.metric("❔ Unknown", unk)

    t1, t2, t3, t4 = st.tabs(
        ["📊 Overview", "🎯 Requirements", "✍️ Rewrite Ideas", "🚀 Action Plan"]
    )

    with t1:
        left, right = st.columns(2)
        with left:
            st.subheader("💪 Strengths")
            for s in d.get("strengths", []) or ["No clear strengths identified."]:
                st.markdown(f"- {s}")
        with right:
            st.subheader("🧩 Gaps")
            for g in d.get("gaps", []) or ["No major gaps found."]:
                st.markdown(f"- {g}")
        st.subheader("🔑 Keywords to consider")
        st.caption("Add these ONLY if you genuinely have that experience.")
        kws = d.get("missing_keywords", []) or []
        if kws:
            st.markdown("".join(f'<span class="tag">{esc(k)}</span>' for k in kws), unsafe_allow_html=True)
        else:
            st.write("Nothing important is missing.")

    with t2:
        if not reqs:
            st.info("No requirement breakdown was returned.")
        for r in reqs:
            st.markdown(
                f"""<div class="item-card">{status_pill(r.get("status"))}
<b>{esc(r.get("requirement"))}</b><br><small>Evidence: {esc(r.get("evidence"))}</small></div>""",
                unsafe_allow_html=True,
            )

    with t3:
        rewrites = d.get("rewrite_suggestions", []) or []
        if not rewrites:
            st.info("No rewrite suggestions were returned.")
        for r in rewrites:
            with st.expander(f"📝 {r.get('section', 'Section')}"):
                st.markdown(f"**Current:** {r.get('current', '')}")
                st.markdown(f"**Suggested:** {r.get('suggested', '')}")
                st.caption(f"Why: {r.get('reason', '')}")
        st.warning(
            "Suggestions only rephrase what is already in your resume. "
            "Replace any [placeholders] with real facts, or leave them out."
        )

    with t4:
        st.subheader("🚀 Your next steps")
        for i, step in enumerate(d.get("next_steps", []) or [], 1):
            st.markdown(f"**{i}.** {step}")
        st.subheader("🧹 Formatting & ATS tips")
        for tip in d.get("formatting_tips", []) or []:
            st.markdown(f"- {tip}")

    st.download_button(
        "⬇️ Download report (.md)",
        data=build_report_markdown(d),
        file_name="resume_review_report.md",
        mime="text/markdown",
        use_container_width=True,
    )


def load_sample():
    st.session_state["resume_text"] = SAMPLE_RESUME
    st.session_state["jd_text"] = SAMPLE_JD
    st.session_state["source"] = "Paste text"


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
api_key, model = get_secrets()

with st.sidebar:
    st.header("⚙️ Settings")
    tone = st.selectbox(
        "Feedback style",
        ["Friendly coach", "Strict recruiter", "Concise bullet points"],
    )
    focus = st.multiselect(
        "Extra focus (optional)",
        ["Action verbs", "Measurable results", "ATS keywords", "Summary section", "Skills section", "Fresher / entry-level"],
    )
    st.divider()
    st.subheader("🧭 How it works")
    st.markdown(
        """
1. Add your resume (PDF or text)
2. Paste the job description
3. Click **Review my resume**
4. Read your score & action plan
"""
    )
    st.divider()
    st.caption(f"Model: `{model}`")
    if api_key:
        st.success("API key found ✅")
    else:
        st.error("API key missing ❌")
    st.button("✨ Try with sample data", on_click=load_sample, use_container_width=True)
    if st.button("🔌 Test Groq connection", use_container_width=True, disabled=not api_key):
        with st.spinner("Checking Groq..."):
            ok, info = test_groq_connection(api_key)
        if ok:
            st.success("Connected to Groq and the key works ✅")
            if model in info:
                st.caption(f"Model `{model}` is available.")
            else:
                st.warning(f"Model `{model}` is NOT in your list. Try `openai/gpt-oss-120b`.")
        else:
            st.error(info)

# ----------------------------------------------------------------------------
# Main page
# ----------------------------------------------------------------------------
st.markdown(
    """
<div class="hero">
  <h1>📄 Resume Review Agent</h1>
  <p>Check how well your resume matches a job - honestly, with no made-up skills - and get a clear plan to improve it.</p>
</div>
""",
    unsafe_allow_html=True,
)

if not api_key:
    st.error(
        "🔑 **GROQ_API_KEY is missing.** Add it in Streamlit Cloud → App settings → Secrets "
        '(or in `.streamlit/secrets.toml` locally), like this: `GROQ_API_KEY = "your_key"`.'
    )

col_a, col_b = st.columns(2, gap="large")

with col_a:
    st.subheader("1️⃣ Your resume")
    source = st.radio("How do you want to add it?", ["Upload PDF", "Paste text"], horizontal=True, key="source")
    resume_text, pdf_error = "", None

    if source == "Upload PDF":
        up = st.file_uploader("Upload your resume (PDF)", type=["pdf"])
        if up is not None:
            resume_text, pdf_error = extract_pdf_text(up)
            if pdf_error:
                st.error(pdf_error)
            else:
                st.success(f"PDF read successfully ({len(resume_text):,} characters).")
                with st.expander("Preview extracted text"):
                    st.text(resume_text[:1500] + ("..." if len(resume_text) > 1500 else ""))
    else:
        resume_text = st.text_area("Paste your resume text", height=320, key="resume_text",
                                   placeholder="Paste your full resume here...")
        st.caption(f"{len(resume_text):,} characters")

with col_b:
    st.subheader("2️⃣ Target job")
    jd_text = st.text_area("Paste the job description", height=320, key="jd_text",
                           placeholder="Paste the job posting here...")
    st.caption(f"{len(jd_text):,} characters")

st.write("")
go = st.button("🔍 Review my resume", type="primary", use_container_width=True, disabled=not api_key)

if go:
    resume_clean, jd_clean = (resume_text or "").strip(), (jd_text or "").strip()
    if source == "Upload PDF" and not resume_clean and not pdf_error:
        st.warning("Please upload your resume PDF first.")
    elif not resume_clean:
        st.warning("Your resume is empty. Please upload a PDF or paste your resume text.")
    elif not jd_clean:
        st.warning("Please paste the job description so we can compare.")
    else:
        if len(resume_clean) > MAX_RESUME_CHARS:
            st.info(f"Your resume is long, so only the first {MAX_RESUME_CHARS:,} characters will be reviewed.")
            resume_clean = resume_clean[:MAX_RESUME_CHARS]
        if len(jd_clean) > MAX_JD_CHARS:
            st.info(f"The job description is long, so only the first {MAX_JD_CHARS:,} characters will be used.")
            jd_clean = jd_clean[:MAX_JD_CHARS]

        try:
            with st.spinner("The agent is reading your resume... this takes 10-40 seconds ⏳"):
                raw = run_review(resume_clean, jd_clean, api_key, model, tone, focus)
            st.session_state["raw"] = raw
            st.session_state["data"] = parse_json_output(raw)
        except Exception as exc:  # noqa: BLE001
            st.session_state.pop("data", None)
            st.session_state.pop("raw", None)
            st.error(friendly_error(exc))
            with st.expander("🔧 Technical details (share this if you need help)"):
                st.code(f"{type(exc).__name__}: {redact(exc, api_key)}")

if st.session_state.get("raw"):
    st.divider()
    st.header("📋 Your Review")
    data = st.session_state.get("data")
    if data:
        show_results(data)
    else:
        st.info("The result came back in plain text format. Here it is:")
        st.markdown(st.session_state["raw"])

st.markdown(
    '<div class="footer-note">Built with CrewAI • Groq • Streamlit &nbsp;|&nbsp; '
    "AI can make mistakes - always double-check suggestions against your real experience.</div>",
    unsafe_allow_html=True,
)
