import os
import streamlit as st
import duckdb
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import pickle
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import warnings
warnings.filterwarnings('ignore')

load_dotenv()

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="FinRisk Terminal",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={}
)

# ── Global CSS — Dark Luxury Bloomberg Style ──────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Playfair+Display:wght@400;600&family=DM+Sans:wght@300;400;500&display=swap');

/* Base */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #080808;
    color: #E8E0D0;
}

/* Main background */
.stApp {
    background: #080808;
}

/* Sidebar — always visible, never collapsible */
[data-testid="stSidebar"] {
    background: #0D0D0D !important;
    border-right: 1px solid #2A2400 !important;
    min-width: 220px !important;
    max-width: 220px !important;
    transform: none !important;
    visibility: visible !important;
}

[data-testid="stSidebar"] * {
    color: #E8E0D0 !important;
}

/* Hide ALL collapse/expand buttons */
[data-testid="collapsedControl"],
button[data-testid="collapsedControl"],
[data-testid="stSidebarCollapseButton"],
button[kind="headerNoPadding"],
.css-1rs6os, .css-17ziqus {
    display: none !important;
    visibility: hidden !important;
}

/* Sidebar inner padding */
[data-testid="stSidebar"] > div:first-child {
    padding: 0 16px !important;
    overflow-y: auto !important;
}

/* Sidebar radio buttons */
[data-testid="stSidebar"] .stRadio > div {
    gap: 0 !important;
}

[data-testid="stSidebar"] .stRadio > div > label {
    font-family: 'DM Mono', monospace !important;
    font-size: 12px !important;
    letter-spacing: 0.08em !important;
    color: #8A7A5A !important;
    text-transform: uppercase !important;
    padding: 12px 8px !important;
    border-bottom: 1px solid #1A1800 !important;
    display: block !important;
    width: 100% !important;
    transition: color 0.2s !important;
    cursor: pointer !important;
}

[data-testid="stSidebar"] .stRadio > div > label:hover {
    color: #C9A84C !important;
}

/* Hide streamlit branding */
#MainMenu, footer, header {visibility: hidden;}

/* Metric cards */
[data-testid="metric-container"] {
    background: #0F0F0F;
    border: 1px solid #2A2400;
    border-top: 2px solid #C9A84C;
    padding: 20px;
    border-radius: 2px;
}

[data-testid="metric-container"] label {
    font-family: 'DM Mono', monospace !important;
    font-size: 10px !important;
    letter-spacing: 0.15em;
    color: #8A7A5A !important;
    text-transform: uppercase;
}

[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'Playfair Display', serif !important;
    font-size: 32px !important;
    color: #C9A84C !important;
}

/* Text input */
.stTextInput > div > div > input {
    background: #0F0F0F !important;
    border: 1px solid #2A2400 !important;
    border-radius: 2px !important;
    color: #E8E0D0 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 13px !important;
    padding: 12px 16px !important;
}

.stTextInput > div > div > input:focus {
    border-color: #C9A84C !important;
    box-shadow: 0 0 0 1px #C9A84C22 !important;
}

/* Buttons */
.stButton > button {
    background: transparent !important;
    border: 1px solid #C9A84C !important;
    color: #C9A84C !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 10px 24px !important;
    border-radius: 2px !important;
    transition: all 0.2s !important;
}

.stButton > button:hover {
    background: #C9A84C !important;
    color: #080808 !important;
}

/* Selectbox */
.stSelectbox > div > div {
    background: #0F0F0F !important;
    border: 1px solid #2A2400 !important;
    border-radius: 2px !important;
    color: #E8E0D0 !important;
    font-family: 'DM Mono', monospace !important;
}

/* Dataframe */
.stDataFrame {
    border: 1px solid #2A2400 !important;
}

/* Spinner */
.stSpinner > div {
    border-top-color: #C9A84C !important;
}

/* Divider */
hr {
    border-color: #1A1A00 !important;
    margin: 2rem 0 !important;
}

/* Success/error boxes */
.stSuccess {
    background: #0A0F08 !important;
    border: 1px solid #2A4A1A !important;
    border-left: 3px solid #5A8A3A !important;
    border-radius: 2px !important;
    color: #A8C888 !important;
}

.stError {
    background: #0F0808 !important;
    border: 1px solid #4A1A1A !important;
    border-left: 3px solid #C94C4C !important;
    border-radius: 2px !important;
}

/* Expander */
.streamlit-expanderHeader {
    background: #0F0F0F !important;
    border: 1px solid #2A2400 !important;
    color: #8A7A5A !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 0.08em;
}

.streamlit-expanderContent {
    background: #0A0A0A !important;
    border: 1px solid #1A1800 !important;
    border-top: none !important;
    color: #8A7A5A !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 12px !important;
}

/* Info boxes used as example questions */
.stInfo {
    background: #0F0D08 !important;
    border: 1px solid #2A2000 !important;
    border-left: 3px solid #C9A84C !important;
    border-radius: 2px !important;
    color: #8A7A5A !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 11px !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #080808; }
::-webkit-scrollbar-thumb { background: #2A2400; border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: #C9A84C; }

/* ── Loading screen ── */
#finrisk-loader {
    position: fixed; inset: 0;
    background: #080808;
    z-index: 99999;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    animation: loaderFadeOut 0.8s ease 2.8s forwards;
}
@keyframes loaderFadeOut {
    to { opacity: 0; pointer-events: none; visibility: hidden; }
}
#finrisk-loader .logo {
    font-family: 'Playfair Display', serif;
    font-size: 48px;
    color: #C9A84C;
    opacity: 0;
    animation: logoReveal 0.8s ease 0.3s forwards;
}
@keyframes logoReveal {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}
#finrisk-loader .tagline {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.3em;
    color: #4A3A1A;
    text-transform: uppercase;
    margin-top: 12px;
    opacity: 0;
    animation: logoReveal 0.8s ease 0.7s forwards;
}
#finrisk-loader .bar-track {
    width: 200px;
    height: 1px;
    background: #1A1800;
    margin-top: 40px;
    overflow: hidden;
    opacity: 0;
    animation: logoReveal 0.4s ease 1s forwards;
}
#finrisk-loader .bar-fill {
    height: 100%;
    width: 0%;
    background: #C9A84C;
    animation: barLoad 1.6s ease 1.1s forwards;
}
@keyframes barLoad { to { width: 100%; } }
#finrisk-loader canvas {
    position: absolute; inset: 0;
    pointer-events: none;
}

/* ── Page fade-in ── */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}
.fade-up-1 { animation: fadeUp 0.6s ease 3.0s both; }
.fade-up-2 { animation: fadeUp 0.6s ease 3.2s both; }
.fade-up-3 { animation: fadeUp 0.6s ease 3.4s both; }
.fade-up-4 { animation: fadeUp 0.6s ease 3.6s both; }

/* ── Metric card hover lift ── */
[data-testid="metric-container"] {
    transition: border-color 0.3s, transform 0.3s !important;
}
[data-testid="metric-container"]:hover {
    border-top-color: #E8C86C !important;
    transform: translateY(-3px) !important;
}

/* ── Chart fade in ── */
.stPlotlyChart { animation: fadeUp 0.7s ease both; }

/* ── Dataframe fade in ── */
.stDataFrame {
    animation: fadeUp 0.7s ease 0.3s both;
    border: 1px solid #2A2400 !important;
}

/* ── Expander hover ── */
.streamlit-expanderHeader {
    transition: background 0.2s, color 0.2s !important;
}
.streamlit-expanderHeader:hover {
    background: #1A1800 !important;
    color: #C9A84C !important;
}

/* ── Button glow on hover ── */
.stButton > button:hover {
    background: #C9A84C !important;
    color: #080808 !important;
    box-shadow: 0 0 16px #C9A84C33 !important;
}

/* ── Sidebar nav gold bar on hover ── */
[data-testid="stSidebar"] .stRadio > div > label {
    position: relative; overflow: hidden;
}
[data-testid="stSidebar"] .stRadio > div > label::before {
    content: '';
    position: absolute;
    left: -4px; top: 0; bottom: 0;
    width: 3px;
    background: #C9A84C;
    transition: left 0.2s ease;
}
[data-testid="stSidebar"] .stRadio > div > label:hover::before {
    left: 0;
}

/* ── Gold shimmer text ── */
@keyframes shimmer {
    0%   { background-position: -200% center; }
    100% { background-position: 200% center; }
}
.gold-shimmer {
    background: linear-gradient(90deg, #8A6A2C 0%, #C9A84C 40%, #E8D08C 50%, #C9A84C 60%, #8A6A2C 100%);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: shimmer 4s linear infinite;
}

/* ── Blinking cursor ── */
@keyframes blink {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0; }
}
.cursor::after {
    content: '|';
    color: #C9A84C;
    animation: blink 1s step-end infinite;
    margin-left: 2px;
}

/* ── Scanning line ── */
@keyframes scanLine {
    0%   { top: -2px; opacity: 0.4; }
    100% { top: 100%; opacity: 0; }
}
.scan-container {
    position: relative;
    overflow: hidden;
}
.scan-line {
    position: absolute;
    left: 0; right: 0;
    height: 1px;
    background: linear-gradient(to right, transparent, #C9A84C55, transparent);
    animation: scanLine 4s ease-in-out infinite;
    pointer-events: none;
}
</style>
""", unsafe_allow_html=True)

# ── Loading screen with particle network ─────────────────
st.markdown("""
<div id="finrisk-loader">
    <canvas id="particle-canvas"></canvas>
    <div class="logo gold-shimmer">◈ FinRisk Terminal</div>
    <div class="tagline">Initialising risk intelligence</div>
    <div class="bar-track"><div class="bar-fill"></div></div>
</div>
<script>
(function(){
    const canvas = document.getElementById('particle-canvas');
    if(!canvas) return;
    const ctx = canvas.getContext('2d');
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    const pts = Array.from({length:55},()=>({
        x: Math.random()*canvas.width,
        y: Math.random()*canvas.height,
        r: Math.random()*1.4+0.3,
        dx:(Math.random()-0.5)*0.35,
        dy:(Math.random()-0.5)*0.35,
        a: Math.random()*0.4+0.1
    }));
    function frame(){
        ctx.clearRect(0,0,canvas.width,canvas.height);
        pts.forEach(p=>{
            ctx.beginPath();
            ctx.arc(p.x,p.y,p.r,0,Math.PI*2);
            ctx.fillStyle=`rgba(201,168,76,${p.a})`;
            ctx.fill();
            p.x+=p.dx; p.y+=p.dy;
            if(p.x<0||p.x>canvas.width)  p.dx*=-1;
            if(p.y<0||p.y>canvas.height) p.dy*=-1;
        });
        for(let i=0;i<pts.length;i++){
            for(let j=i+1;j<pts.length;j++){
                const d=Math.hypot(pts[i].x-pts[j].x,pts[i].y-pts[j].y);
                if(d<130){
                    ctx.beginPath();
                    ctx.moveTo(pts[i].x,pts[i].y);
                    ctx.lineTo(pts[j].x,pts[j].y);
                    ctx.strokeStyle=`rgba(201,168,76,${0.07*(1-d/130)})`;
                    ctx.lineWidth=0.5;
                    ctx.stroke();
                }
            }
        }
        const loader=document.getElementById('finrisk-loader');
        if(loader && getComputedStyle(loader).visibility!=='hidden') requestAnimationFrame(frame);
    }
    frame();
})();
</script>
""", unsafe_allow_html=True)

# ── Plotly dark luxury theme ──────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor='#080808',
    plot_bgcolor='#0A0A0A',
    font=dict(family='DM Mono, monospace', color='#8A7A5A', size=11),
    title_font=dict(family='Playfair Display, serif', color='#E8E0D0', size=16),
    xaxis=dict(
        gridcolor='#1A1800', gridwidth=0.5,
        linecolor='#2A2400', tickcolor='#2A2400',
        tickfont=dict(color='#8A7A5A', size=10)
    ),
    yaxis=dict(
        gridcolor='#1A1800', gridwidth=0.5,
        linecolor='#2A2400', tickcolor='#2A2400',
        tickfont=dict(color='#8A7A5A', size=10)
    ),
    margin=dict(l=16, r=16, t=48, b=16),
    legend=dict(
        bgcolor='#0F0F0F',
        bordercolor='#2A2400',
        borderwidth=1,
        font=dict(color='#8A7A5A', size=10)
    ),
    transition=dict(
        duration=800,
        easing='cubic-in-out'
    )
)

GOLD       = '#C9A84C'
GOLD_DIM   = '#8A6A2C'
RED        = '#C94C4C'
GREEN      = '#5A8A3A'
AMBER      = '#C97A2C'
BG_CARD    = '#0F0F0F'
BORDER     = '#2A2400'

# ── Paths — portable, works locally and on cloud ──────────
BASE_DIR    = Path(__file__).parent.parent.parent
DB_PATH     = str(BASE_DIR / "data" / "financial_warehouse.duckdb")
CHROMA_PATH = str(BASE_DIR / "data" / "chroma_db")
MODEL_PATH  = str(BASE_DIR / "models" / "xgboost_risk_model.pkl")
SCALER_PATH = str(BASE_DIR / "models" / "scaler.pkl")

# ── Cached loaders ────────────────────────────────────────
@st.cache_resource
def load_db():
    return duckdb.connect(DB_PATH, read_only=True)

@st.cache_resource
def load_model():
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

@st.cache_resource
def load_rag_chain():
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )
    prompt_template = """You are a senior financial risk analyst.
Use the following SEC filing excerpts to answer the question with precision.
Always cite which company and filing date the information comes from.
If information is insufficient, state that clearly.

SEC Filing Context:
{context}

Analyst Query: {question}

Analysis:"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    # ── LLM: Groq (replaces Ollama — no local model needed) ──
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.1,
        max_tokens=1024,
        api_key=os.getenv("GROQ_API_KEY"),
    )

    def format_docs(docs):
        return "\n\n".join([
            f"[{doc.metadata['ticker']} | {doc.metadata['filing_date']}]\n{doc.page_content}"
            for doc in docs
        ])

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )
    return chain, retriever

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style='padding: 24px 0 32px 0;'>
        <div style='font-family: "Playfair Display", serif; font-size: 22px; color: {GOLD}; letter-spacing: 0.02em;'>◈ FinRisk</div>
        <div style='font-family: "DM Mono", monospace; font-size: 10px; color: #4A3A1A; letter-spacing: 0.2em; text-transform: uppercase; margin-top: 4px;'>Terminal v1.0</div>
        <div style='margin-top: 20px; height: 1px; background: linear-gradient(to right, {GOLD}, transparent);'></div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "",
        ["DASHBOARD", "AI ANALYST", "EQUITY LENS"],
        label_visibility="collapsed"
    )

    st.markdown(f"""
    <div style='margin-top: 40px; padding-top: 16px; border-top: 1px solid #1A1800;'>
        <div style='font-family: "DM Mono", monospace; font-size: 9px; color: #3A2A0A; letter-spacing: 0.15em; text-transform: uppercase; line-height: 2;'>
            XGBoost · RAG · LangChain<br>
            DuckDB · ChromaDB · Groq
        </div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ══════════════════════════════════════════════════════════
if page == "DASHBOARD":

    st.markdown(f"""
    <div class='scan-container fade-up-1' style='padding: 8px 0 32px 0;'>
        <div class='scan-line'></div>
        <div style='font-family: "DM Mono", monospace; font-size: 10px; color: {GOLD_DIM}; letter-spacing: 0.2em; text-transform: uppercase; margin-bottom: 8px;'>Live Risk Intelligence</div>
        <div style='font-family: "Playfair Display", serif; font-size: 36px; color: #E8E0D0; font-weight: 400; line-height: 1.1;' class='cursor'>Risk Dashboard</div>
        <div style='font-family: "DM Sans", sans-serif; font-size: 14px; color: #4A3A1A; margin-top: 8px;'>XGBoost-powered risk scoring across S&P 500 equities · SEC filing analysis</div>
    </div>
    """, unsafe_allow_html=True)

    try:
        con = load_db()

        latest = con.execute("""
            SELECT ticker, date, close, risk_score, risk_category, volatility_30d
            FROM risk_scores
            WHERE date = (SELECT MAX(date) FROM risk_scores)
            ORDER BY risk_score DESC
        """).df()

        high   = len(latest[latest['risk_category'] == 'High Risk'])
        medium = len(latest[latest['risk_category'] == 'Medium Risk'])
        low    = len(latest[latest['risk_category'] == 'Low Risk'])
        avg_score = latest['risk_score'].mean()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("HIGH RISK", high,   delta=None)
        c2.metric("MEDIUM RISK", medium, delta=None)
        c3.metric("LOW RISK", low,    delta=None)
        c4.metric("AVG SCORE", f"{avg_score:.2f}", delta=None)

        st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

        colors = latest['risk_category'].map({
            'High Risk':   RED,
            'Medium Risk': AMBER,
            'Low Risk':    GREEN
        })

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=latest['ticker'],
            y=latest['risk_score'].round(2),
            marker=dict(color=colors, line=dict(color='#1A1800', width=0.5)),
            text=[f"{v:.2f}" for v in latest['risk_score']],
            textposition='outside',
            textfont=dict(family='DM Mono, monospace', size=10, color='#8A7A5A'),
        ))
        fig.add_hline(y=0.6, line_dash="dot", line_color=AMBER, line_width=1,
                      annotation_text="MEDIUM THRESHOLD",
                      annotation_font=dict(size=9, color=AMBER, family='DM Mono, monospace'))
        fig.add_hline(y=0.3, line_dash="dot", line_color=GREEN, line_width=1,
                      annotation_text="LOW THRESHOLD",
                      annotation_font=dict(size=9, color=GREEN, family='DM Mono, monospace'))
        fig.update_layout(
            **PLOTLY_LAYOUT,
            title="CURRENT RISK SCORES — S&P 500 SELECTION",
            yaxis_range=[0, 1.1],
            showlegend=False,
            height=380,
            bargap=0.35,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"""
        <div style='display:flex; gap:24px; margin-bottom:24px; font-family:"DM Mono",monospace; font-size:10px; letter-spacing:0.1em;'>
            <span style='color:{RED};'>▮ HIGH RISK  &gt;0.60</span>
            <span style='color:{AMBER};'>▮ MEDIUM RISK  0.30–0.60</span>
            <span style='color:{GREEN};'>▮ LOW RISK  &lt;0.30</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style='font-family:"DM Mono",monospace; font-size:10px; color:{GOLD_DIM}; letter-spacing:0.15em; text-transform:uppercase; margin-bottom:12px;'>
            ◈ Risk Scorecard
        </div>
        """, unsafe_allow_html=True)

        display_df = latest[['ticker','close','risk_score','risk_category','volatility_30d']].copy()
        display_df.columns = ['TICKER', 'PRICE ($)', 'RISK SCORE', 'CATEGORY', '30D VOLATILITY']
        display_df['PRICE ($)'] = display_df['PRICE ($)'].round(2)
        display_df['RISK SCORE'] = display_df['RISK SCORE'].round(4)
        display_df['30D VOLATILITY'] = display_df['30D VOLATILITY'].round(4)
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Database error: {e}")

# ══════════════════════════════════════════════════════════
# PAGE 2 — AI ANALYST
# ══════════════════════════════════════════════════════════
elif page == "AI ANALYST":

    st.markdown(f"""
    <div style='padding: 8px 0 32px 0;'>
        <div style='font-family: "DM Mono", monospace; font-size: 10px; color: {GOLD_DIM}; letter-spacing: 0.2em; text-transform: uppercase; margin-bottom: 8px;'>Powered by RAG + Groq (Llama 3)</div>
        <div style='font-family: "Playfair Display", serif; font-size: 36px; color: #E8E0D0; font-weight: 400; line-height: 1.1;'>AI Risk Analyst</div>
        <div style='font-family: "DM Sans", sans-serif; font-size: 14px; color: #4A3A1A; margin-top: 8px;'>Natural language queries over real SEC 10-K filings · Source-grounded answers</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style='font-family:"DM Mono",monospace; font-size:10px; color:{GOLD_DIM}; letter-spacing:0.15em; text-transform:uppercase; margin-bottom:12px;'>
        ◈ Sample Queries
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div style='background:{BG_CARD}; border:1px solid {BORDER}; border-left:2px solid {GOLD}; padding:12px 16px; border-radius:2px; font-family:"DM Mono",monospace; font-size:11px; color:#8A7A5A; line-height:1.6;'>
            What are Tesla's main risk factors?
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div style='background:{BG_CARD}; border:1px solid {BORDER}; border-left:2px solid {GOLD}; padding:12px 16px; border-radius:2px; font-family:"DM Mono",monospace; font-size:11px; color:#8A7A5A; line-height:1.6;'>
            How does JPMorgan manage credit risk?
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div style='background:{BG_CARD}; border:1px solid {BORDER}; border-left:2px solid {GOLD}; padding:12px 16px; border-radius:2px; font-family:"DM Mono",monospace; font-size:11px; color:#8A7A5A; line-height:1.6;'>
            Which companies mention inflation as a risk?
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    st.markdown(f"""
    <div style='font-family:"DM Mono",monospace; font-size:10px; color:{GOLD_DIM}; letter-spacing:0.15em; text-transform:uppercase; margin-bottom:8px;'>
        ◈ Enter Analyst Query
    </div>
    """, unsafe_allow_html=True)

    question = st.text_input(
        "",
        placeholder="e.g. What are Apple's supply chain risks?",
        label_visibility="collapsed"
    )

    col_btn, col_space = st.columns([1, 4])
    with col_btn:
        submit = st.button("◈ RUN ANALYSIS", type="primary")

    if submit and question:
        with st.spinner("Scanning SEC filings..."):
            try:
                chain, retriever = load_rag_chain()
                answer  = chain.invoke(question)
                sources = retriever.invoke(question)

                st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
                st.markdown(f"""
                <div style='background:{BG_CARD}; border:1px solid {BORDER}; border-top:2px solid {GOLD}; padding:24px; border-radius:2px; margin-bottom:20px;'>
                    <div style='font-family:"DM Mono",monospace; font-size:9px; color:{GOLD}; letter-spacing:0.2em; text-transform:uppercase; margin-bottom:16px;'>◈ Analysis Output</div>
                    <div style='font-family:"DM Sans",sans-serif; font-size:14px; color:#C8C0B0; line-height:1.8;'>{answer}</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                <div style='font-family:"DM Mono",monospace; font-size:10px; color:{GOLD_DIM}; letter-spacing:0.15em; text-transform:uppercase; margin-bottom:12px;'>
                    ◈ Source Documents
                </div>
                """, unsafe_allow_html=True)

                for doc in sources:
                    ticker = doc.metadata['ticker']
                    date   = doc.metadata['filing_date']
                    with st.expander(f"  {ticker}  ·  10-K  ·  {date}"):
                        st.markdown(f"""
                        <div style='font-family:"DM Mono",monospace; font-size:11px; color:#5A4A2A; line-height:1.7;'>
                            {doc.page_content[:600]}...
                        </div>
                        """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Analysis failed: {e}. Make sure GROQ_API_KEY is set in your .env file.")

# ══════════════════════════════════════════════════════════
# PAGE 3 — EQUITY LENS
# ══════════════════════════════════════════════════════════
elif page == "EQUITY LENS":

    st.markdown(f"""
    <div style='padding: 8px 0 32px 0;'>
        <div style='font-family: "DM Mono", monospace; font-size: 10px; color: {GOLD_DIM}; letter-spacing: 0.2em; text-transform: uppercase; margin-bottom: 8px;'>Single Stock Deep Dive</div>
        <div style='font-family: "Playfair Display", serif; font-size: 36px; color: #E8E0D0; font-weight: 400; line-height: 1.1;'>Equity Lens</div>
        <div style='font-family: "DM Sans", sans-serif; font-size: 14px; color: #4A3A1A; margin-top: 8px;'>Price history · Risk trajectory · Volatility surface</div>
    </div>
    """, unsafe_allow_html=True)

    tickers = ['AAPL','MSFT','JPM','BAC','GS','AMZN','TSLA','XOM','JNJ','WMT']

    st.markdown(f"""
    <div style='font-family:"DM Mono",monospace; font-size:10px; color:{GOLD_DIM}; letter-spacing:0.15em; text-transform:uppercase; margin-bottom:8px;'>
        ◈ Select Equity
    </div>
    """, unsafe_allow_html=True)

    ticker = st.selectbox("", tickers, label_visibility="collapsed")

    try:
        con = load_db()
        df = con.execute(f"""
            SELECT date, close, daily_return, volatility_30d, risk_score, risk_category
            FROM risk_scores
            WHERE ticker = '{ticker}'
            ORDER BY date
        """).df()

        latest_row = df.iloc[-1]
        cat        = str(latest_row['risk_category'])
        score      = latest_row['risk_score']
        price      = latest_row['close']
        vol        = latest_row['volatility_30d']

        cat_color = {'High Risk': RED, 'Medium Risk': AMBER, 'Low Risk': GREEN}.get(cat, GOLD)

        st.markdown(f"""
        <div style='background:{BG_CARD}; border:1px solid {BORDER}; border-left:3px solid {cat_color}; padding:20px 24px; border-radius:2px; margin-bottom:24px; display:flex; justify-content:space-between; align-items:center;'>
            <div>
                <div style='font-family:"Playfair Display",serif; font-size:28px; color:#E8E0D0;'>{ticker}</div>
                <div style='font-family:"DM Mono",monospace; font-size:10px; color:#4A3A1A; letter-spacing:0.15em; margin-top:4px;'>NYSE · EQUITY · S&P 500</div>
            </div>
            <div style='text-align:right;'>
                <div style='font-family:"Playfair Display",serif; font-size:28px; color:{GOLD};'>${price:.2f}</div>
                <div style='font-family:"DM Mono",monospace; font-size:10px; color:{cat_color}; letter-spacing:0.1em; margin-top:4px;'>● {cat.upper()} · {score:.3f}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("RISK SCORE", f"{score:.3f}")
        c2.metric("CATEGORY",   cat.upper().replace(' RISK',''))
        c3.metric("30D VOL",    f"{vol:.4f}")
        c4.metric("LAST CLOSE", f"${price:.2f}")

        st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=df['date'], y=df['close'].round(2),
            mode='lines', name='Close Price',
            line=dict(color=GOLD, width=1.5),
            fill='tozeroy', fillcolor='rgba(201,168,76,0.04)'
        ))
        fig1.update_layout(**PLOTLY_LAYOUT, title=f"{ticker} — PRICE HISTORY",
                           height=280, showlegend=False, yaxis_title="USD")
        st.plotly_chart(fig1, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=df['date'], y=df['risk_score'].round(4),
                mode='lines', name='Risk Score',
                line=dict(color=RED, width=1.5),
            ))
            fig2.add_hrect(y0=0.6, y1=1.0, fillcolor=RED,   opacity=0.04, line_width=0)
            fig2.add_hrect(y0=0.3, y1=0.6, fillcolor=AMBER,  opacity=0.04, line_width=0)
            fig2.add_hrect(y0=0.0, y1=0.3, fillcolor=GREEN,  opacity=0.04, line_width=0)
            fig2.add_hline(y=0.6, line_dash="dot", line_color=AMBER, line_width=0.8)
            fig2.add_hline(y=0.3, line_dash="dot", line_color=GREEN, line_width=0.8)
            fig2.update_layout(**PLOTLY_LAYOUT, title="RISK SCORE TRAJECTORY",
                               height=260, showlegend=False, yaxis_range=[0, 1])
            st.plotly_chart(fig2, use_container_width=True)

        with col2:
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(
                x=df['date'], y=df['volatility_30d'].round(4),
                mode='lines', name='30D Volatility',
                line=dict(color=AMBER, width=1.5),
                fill='tozeroy', fillcolor='rgba(201,122,44,0.06)'
            ))
            fig3.update_layout(**PLOTLY_LAYOUT, title="30-DAY ROLLING VOLATILITY",
                               height=260, showlegend=False)
            st.plotly_chart(fig3, use_container_width=True)

        fig4 = go.Figure()
        fig4.add_trace(go.Histogram(
            x=df['daily_return'].dropna().round(4),
            nbinsx=60,
            marker=dict(color=GOLD_DIM, line=dict(color='#1A1800', width=0.3)),
            name='Daily Return'
        ))
        fig4.update_layout(**PLOTLY_LAYOUT, title=f"{ticker} — DAILY RETURN DISTRIBUTION",
                           height=240, showlegend=False,
                           xaxis_title="Daily Return", yaxis_title="Frequency")
        st.plotly_chart(fig4, use_container_width=True)

    except Exception as e:
        st.error(f"Data error: {e}")