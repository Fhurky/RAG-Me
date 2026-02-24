import os
import streamlit as st
from main import load_cv, CVSearchTool, create_agents, ask

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Furkan KOÇAL · AI Twin",
    page_icon="⚡",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

/* ---------- global ---------- */
html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}

.stApp {
    background: #0d0d0d;
    color: #e8e8e8;
}

/* ---------- hide streamlit chrome ---------- */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 6rem; max-width: 760px; }

/* ---------- hero header ---------- */
.hero {
    border: 1px solid #2a2a2a;
    border-radius: 16px;
    padding: 2rem 2.4rem 1.6rem;
    margin-bottom: 1.6rem;
    background: linear-gradient(135deg, #111 0%, #1a1a1a 100%);
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, #00ff8844 0%, transparent 70%);
    pointer-events: none;
}
.hero-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.18em;
    color: #00ff88;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.hero-name {
    font-size: 2rem;
    font-weight: 800;
    color: #f0f0f0;
    margin: 0 0 0.3rem 0;
    letter-spacing: -0.02em;
}
.hero-sub {
    font-size: 0.85rem;
    color: #666;
    font-family: 'Space Mono', monospace;
}
.hero-sub span { color: #00ff88; }

/* ---------- chip tags ---------- */
.chip-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
    margin-top: 1rem;
}
.chip {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    padding: 3px 10px;
    border-radius: 999px;
    border: 1px solid #2a2a2a;
    color: #888;
    background: #141414;
    letter-spacing: 0.05em;
}

/* ---------- chat messages ---------- */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0.3rem 0 !important;
}

/* user bubble */
[data-testid="stChatMessage"][data-testid*="user"],
.stChatMessage:has([aria-label="user"]) {
    justify-content: flex-end;
}

/* ---------- chat input ---------- */
[data-testid="stChatInput"] {
    border-top: 1px solid #1e1e1e !important;
    background: #0d0d0d !important;
    padding-top: 0.8rem !important;
}
[data-testid="stChatInputTextArea"] {
    background: #161616 !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 12px !important;
    color: #e8e8e8 !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 0.9rem !important;
}
[data-testid="stChatInputTextArea"]:focus {
    border-color: #00ff88 !important;
    box-shadow: 0 0 0 2px #00ff8822 !important;
}

/* ---------- spinner ---------- */
.stSpinner > div { border-top-color: #00ff88 !important; }

/* ---------- status badge ---------- */
.status-row {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 1.4rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: #555;
}
.status-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: #00ff88;
    box-shadow: 0 0 6px #00ff88;
    animation: pulse 2s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.5; transform: scale(0.85); }
}

/* ---------- divider ---------- */
hr { border-color: #1e1e1e !important; margin: 1rem 0 !important; }

/* ---------- error ---------- */
.stAlert { border-radius: 10px !important; font-family: 'Space Mono', monospace !important; font-size: 0.8rem !important; }

/* ---------- expander ---------- */
.streamlit-expanderHeader {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.75rem !important;
    color: #555 !important;
}

/* ---------- social & cv links ---------- */
.link-row {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.5rem;
    margin-top: 1.2rem;
}
.social-link {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    padding: 5px 12px;
    border-radius: 8px;
    border: 1px solid #2a2a2a;
    color: #888;
    background: #141414;
    text-decoration: none !important;
    letter-spacing: 0.04em;
    transition: border-color 0.2s, color 0.2s, background 0.2s;
}
.social-link:hover {
    color: #e8e8e8;
    border-color: #444;
    background: #1e1e1e;
    text-decoration: none !important;
}
.social-link.linkedin:hover { border-color: #0a66c2; color: #0a8cf0; }
.social-link.github:hover   { border-color: #6e5494; color: #c9b5f5; }
.social-link.gmail:hover    { border-color: #ea4335; color: #ff6b6b; }
.cv-link {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    padding: 5px 14px;
    border-radius: 8px;
    border: 1px solid #00ff8855;
    color: #00ff88;
    background: #00ff8810;
    text-decoration: none !important;
    letter-spacing: 0.04em;
    transition: background 0.2s, border-color 0.2s, box-shadow 0.2s;
    margin-left: auto;
}
.cv-link:hover {
    background: #00ff8822;
    border-color: #00ff88;
    box-shadow: 0 0 10px #00ff8833;
    text-decoration: none !important;
    color: #00ff88;
}
</style>
""", unsafe_allow_html=True)

# ── API key check ─────────────────────────────────────────────────────────────
if not os.getenv("GEMINI_API_KEY"):
    st.error("⚠️  GEMINI_API_KEY bulunamadı. Settings → Variables and secrets kısmına eklemelisin.")
    st.stop()

# ── Links (kendi linklerinle değiştir) ───────────────────────────────────────
LINKEDIN_URL = "https://www.linkedin.com/in/fhurkhan"
GITHUB_URL   = "https://github.com/Fhurky"
GMAIL        = "mailto:furkocal@gmail.com"
CV_GDRIVE    = "https://drive.google.com/file/d/12OEYYj-qAdBq3KmC1T3fCXNf48VFOHGE/view?usp=sharing"

# ── Hero header ───────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
    <div class="hero-label">⚡ AI Digital Twin</div>
    <div class="hero-name">Furkan KOÇAL</div>
    <div class="hero-sub">RAG · ChromaDB · AI Agents &nbsp;|&nbsp; <span>Aktif</span></div>
    <div class="chip-row">
        <span class="chip">Machine Learning</span>
        <span class="chip">Computer Vision</span>
        <span class="chip">Deep Learning</span>
        <span class="chip">LLM</span>
        <span class="chip">Researching</span>
    </div>
    <div class="link-row">
        <a class="social-link linkedin" href="{LINKEDIN_URL}" target="_blank">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor">
                <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433a2.062 2.062 0 01-2.063-2.065 2.064 2.064 0 112.063 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/>
            </svg>
            LinkedIn
        </a>
        <a class="social-link github" href="{GITHUB_URL}" target="_blank">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor">
                <path d="M12 0C5.374 0 0 5.373 0 12c0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23A11.509 11.509 0 0112 5.803c1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576C20.566 21.797 24 17.3 24 12c0-6.627-5.373-12-12-12z"/>
            </svg>
            GitHub
        </a>
        <a class="social-link gmail" href="{GMAIL}">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor">
                <path d="M24 5.457v13.909c0 .904-.732 1.636-1.636 1.636h-3.819V11.73L12 16.64l-6.545-4.91v9.273H1.636A1.636 1.636 0 010 19.366V5.457c0-2.023 2.309-3.178 3.927-1.964L5.455 4.64 12 9.548l6.545-4.908 1.528-1.147C21.69 2.28 24 3.434 24 5.457z"/>
            </svg>
            Gmail
        </a>
        <a class="cv-link" href="{CV_GDRIVE}" target="_blank">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor">
                <path d="M12 16l-5-5 1.41-1.41L11 13.17V4h2v9.17l2.59-2.58L17 11l-5 5zm-7 2h14v2H5v-2z"/>
            </svg>
            Download CV/Resume
        </a>
    </div>
</div>
""", unsafe_allow_html=True)

# ── System init ───────────────────────────────────────────────────────────────
CV_PATH = os.path.join(os.path.dirname(__file__), "CV.txt")

@st.cache_resource(show_spinner=False)
def init_system():
    vector_db = load_cv(
        cv_path=CV_PATH,
        collection_name="furkan_cv",
        persist_directory="./chroma_furkan_cv",
    )
    cv_tool = CVSearchTool(vector_db=vector_db)
    researcher, digital_twin = create_agents(cv_tool)
    return cv_tool, researcher, digital_twin

with st.spinner("Sistem başlatılıyor…"):
    cv_tool, researcher, digital_twin = init_system()

# ── Status badge ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="status-row">
    <div class="status-dot"></div>
    <span>Sistem hazır · CV yüklendi · Sorularınızı bekliyorum</span>
</div>
""", unsafe_allow_html=True)

# ── Chat state ────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Example questions (only when no messages yet) ─────────────────────────────
EXAMPLE_QUESTIONS = [
    "Kendini kısaca tanıtır mısın?",
    "Huawei'deki rolün neydi?",
    "Hangi teknolojileri kullanıyorsun?",
    "Sertifikaların neler?",
]

if not st.session_state.messages:
    cols = st.columns(2)
    for i, q in enumerate(EXAMPLE_QUESTIONS):
        with cols[i % 2]:
            if st.button(q, key=f"eq_{i}", use_container_width=True):
                st.session_state._prefill = q
                st.rerun()

st.divider()

# ── Show chat history ─────────────────────────────────────────────────────────
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# ── Handle prefilled question from buttons ────────────────────────────────────
_prefill = st.session_state.pop("_prefill", None)

# ── Chat input ────────────────────────────────────────────────────────────────
prompt = st.chat_input("Bir şeyler sor…") or _prefill

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Düşünüyorum…"):
            try:
                answer = ask(prompt, researcher, digital_twin, cv_tool)
            except Exception as e:
                st.error(f"Hata oluştu: {e}")
                st.stop()
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

# ── Footer ────────────────────────────────────────────────────────────────────
if st.session_state.messages:
    st.markdown("""
    <div style="text-align:center; margin-top:2rem; font-family:'Space Mono',monospace;
                font-size:0.65rem; color:#333;">
        Powered by Gemini · ChromaDB · CrewAI
    </div>
    """, unsafe_allow_html=True)