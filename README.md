# ◈ FinRisk Terminal
### AI-Powered Financial Risk Intelligence Platform

![Python](https://img.shields.io/badge/Python-3.11-gold?style=flat-square&logo=python&logoColor=white&labelColor=080808&color=C9A84C)
![XGBoost](https://img.shields.io/badge/XGBoost-Risk_Model-gold?style=flat-square&labelColor=080808&color=C9A84C)
![LangChain](https://img.shields.io/badge/LangChain-RAG_Pipeline-gold?style=flat-square&labelColor=080808&color=C9A84C)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-gold?style=flat-square&labelColor=080808&color=C9A84C)
![MLflow](https://img.shields.io/badge/MLflow-Experiment_Tracking-gold?style=flat-square&labelColor=080808&color=C9A84C)
![Groq](https://img.shields.io/badge/LLM-Groq_Llama3-gold?style=flat-square&labelColor=080808&color=C9A84C)

---

## Overview

**FinRisk Terminal** is a production-grade, end-to-end data science project that combines machine learning, generative AI, and real-time data engineering to deliver financial risk intelligence across S&P 500 equities.

The platform ingests **5 years of market data** and **real SEC 10-K filings**, trains an **XGBoost risk scoring model**, and powers an **LLM-based AI copilot** that answers natural language questions grounded in actual company disclosures — all served through a Bloomberg-inspired dark luxury dashboard.

> Built to demonstrate 2026-relevant data science skills: RAG pipelines, MLOps, production deployment, and end-to-end system design.

---

## Live Demo

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://financial-risk-copilot.streamlit.app)

### Three core modules:

| Module | Description |
|---|---|
| 📊 **Risk Dashboard** | Real-time XGBoost risk scores for 10 S&P 500 companies with interactive charts |
| 🤖 **AI Risk Analyst** | RAG-powered copilot answering questions from real SEC 10-K filings |
| 📈 **Equity Lens** | Deep-dive per-stock analysis: price history, risk trajectory, volatility surface |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    FinRisk Terminal                      │
├──────────────┬──────────────────┬───────────────────────┤
│   Data Layer │   ML Layer       │   GenAI Layer         │
│              │                  │                       │
│  yfinance    │  XGBoost         │  LangChain            │
│  SEC EDGAR   │  scikit-learn    │  ChromaDB (vectors)   │
│  DuckDB      │  MLflow          │  HuggingFace embed    │
│  SQL pipeln  │  Evidently       │  Groq (Llama 3)       │
└──────────────┴──────────────────┴───────────────────────┘
                        │
                 Streamlit App
```

---

## Key Features

- **End-to-end ML pipeline** — from raw market data to deployed risk scores
- **RAG over SEC filings** — LLM answers grounded in real 10-K disclosures with source citations
- **Walk-forward time series validation** — no data leakage, production-realistic evaluation
- **MLflow experiment tracking** — full model registry with parameters and metrics logged
- **DuckDB SQL warehouse** — analytical queries over 15,000+ rows of financial data
- **Groq-powered LLM** — fast cloud inference via Llama 3, deployable without local GPU
- **Bloomberg-inspired UI** — dark luxury aesthetic with particle animations and gold accents

---

## Impact & Use Case

FinRisk Terminal is designed for portfolio managers and equity analysts who need risk signals they can actually explain to stakeholders — not just a probability score from a black box. The XGBoost model produces interpretable risk categories (High / Medium / Low) backed by the features that drove each score, while the RAG copilot lets analysts ask plain-English questions and receive answers grounded directly in SEC 10-K filings rather than model hallucinations. This matters in practice: a risk assessment that cites the exact paragraph from Apple's annual report discussing supply chain concentration is far more defensible in an investment committee than one generated from parametric assumptions. The system covers the full analytical workflow — macro risk screening on the dashboard, document-grounded qualitative analysis via the AI copilot, and deep per-stock quantitative review in Equity Lens — making it a self-contained risk intelligence workbench rather than a single-purpose model output.

---

## Tech Stack

### Data Engineering
| Tool | Purpose |
|---|---|
| `DuckDB` | Local analytical SQL data warehouse |
| `yfinance` | Market price data (5 years, 10 tickers) |
| `sec-edgar-downloader` | Real SEC 10-K filing ingestion |
| `pandas` / `SQLAlchemy` | Data transformation pipelines |

### Machine Learning
| Tool | Purpose |
|---|---|
| `XGBoost` | Financial risk scoring model |
| `scikit-learn` | Preprocessing, walk-forward validation |
| `MLflow` | Experiment tracking and model registry |
| `Evidently` | Data drift and model monitoring |

### Generative AI / RAG
| Tool | Purpose |
|---|---|
| `LangChain` | RAG pipeline orchestration (LCEL) |
| `ChromaDB` | Vector store for SEC filing embeddings |
| `HuggingFace` | `all-MiniLM-L6-v2` embedding model |
| `Groq` | Cloud LLM inference (Llama 3 8B) |

### Deployment
| Tool | Purpose |
|---|---|
| `Streamlit` | Interactive dashboard frontend |
| `Docker` | Containerisation |

---

## Model Performance

| Metric | Value |
|---|---|
| ROC-AUC Score | **0.68** |
| Accuracy | **92%** |
| Validation Strategy | Walk-forward time series split |
| Training Data | 12,008 rows (80% split) |
| Test Data | 3,002 rows (20% split) |

> Note: High Risk recall is intentionally conservative — financial risk models prioritise precision to avoid false alarms.

### Feature Importance
Top predictors: `volatility_30d`, `price_vs_ma30`, `daily_return`, `volatility_7d`

---

## Project Structure

```
financial-risk-copilot/
│
├── notebook/
│   ├── 01_data_ingestion.ipynb        # Market data + DuckDB warehouse
│   ├── 02_sec_filings_ingestion.ipynb # SEC 10-K filing download + extraction
│   ├── 03_rag_pipeline.ipynb          # LangChain RAG + ChromaDB + Groq
│   └── 04_risk_scoring_model.ipynb    # XGBoost + MLflow + risk scores
│
├── src/
│   └── api/
│       └── app.py                     # Streamlit dashboard
│
├── data/
│   ├── financial_warehouse.duckdb     # DuckDB warehouse
│   ├── processed/sec_text/            # Extracted SEC filing text
│   └── chroma_db/                     # ChromaDB vector store
│
├── models/
│   ├── xgboost_risk_model.pkl         # Trained XGBoost model
│   └── scaler.pkl                     # Feature scaler
│
├── asset/                             # Screenshots and demo GIFs
├── mlruns/                            # MLflow experiment logs
├── requirements.txt
├── .env                               # Local secrets (never committed)
└── README.md
```

---

## Getting Started

### Prerequisites
- Python 3.11
- Anaconda / Miniconda
- Groq API key — free at [console.groq.com](https://console.groq.com)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/MuntazirAliM/financial-risk-copilot.git
cd financial-risk-copilot

# 2. Create conda environment
conda create -n finrisk python=3.11 -y
conda activate finrisk

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your Groq API key
# Create a .env file in the project root:
# GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxx

# 5. Run the data pipelines (in order)
jupyter lab
# Run notebooks 01 → 02 → 03 → 04 in sequence

# 6. Launch the dashboard
streamlit run src/api/app.py
```

### Environment Variables

Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_key_here      # Required — get free at console.groq.com
HF_TOKEN=your_token_here             # Optional — for higher HuggingFace rate limits
```

---

## Data Sources

- **Market Data**: Yahoo Finance via `yfinance` — daily OHLCV for 10 S&P 500 tickers (2019–2024)
- **SEC Filings**: EDGAR via `sec-edgar-downloader` — last 3 years of 10-K annual reports
- **Tickers**: AAPL, MSFT, JPM, BAC, GS, AMZN, TSLA, XOM, JNJ, WMT

---

## Known Limitations & What I'd Do Differently

**Answer quality is bounded by the model.** Llama 3 8B via Groq is fast and free, and good enough for structured retrieval tasks. For nuanced qualitative risk analysis — reading between the lines of MD&A sections, identifying hedging language, picking up on disclosure tone shifts — a GPT-4 class model would produce materially better outputs. The architecture supports swapping models with a one-line change; the bottleneck is cost, not code.

**Ten tickers is a proof of concept, not a product.** The current pipeline ingests a fixed list of S&P 500 companies defined at notebook-run time. A production version would need dynamic ticker ingestion — a scheduled pipeline that accepts any ticker, downloads the latest 10-K automatically, re-embeds into ChromaDB, and retrains or fine-tunes the risk model incrementally. That's a non-trivial engineering problem I deliberately scoped out to keep the project completable.

**ChromaDB is local and single-node.** It works well for 30 documents and a single user. At scale — thousands of filings, concurrent analysts — I'd replace it with a managed vector database (Pinecone or Weaviate) with proper indexing, access control, and SLAs. The LangChain abstraction makes this a configuration change rather than a rewrite.

**0.68 AUC is the honest ceiling for this data.** Walk-forward validation is the right methodology here — it mirrors how a model would actually be used in production, where you train on history and predict forward. The 0.68 result reflects the genuine difficulty of predicting financial risk from price-derived features alone. Adding alternative data — earnings call transcripts, analyst sentiment, options market implied volatility — would likely push this meaningfully higher, but that data is expensive or requires significant scraping infrastructure.

**I'd add confidence intervals to the risk score.** Currently the model outputs a point estimate (e.g. 0.82 risk score). In practice, a portfolio manager wants to know whether that 0.82 is stable or whether it swings between 0.65 and 0.95 across bootstrap samples. I'd implement conformal prediction intervals on top of the XGBoost output — a technically straightforward addition that would significantly improve the output's usefulness for actual decision-making.

---

## Author

**Muntazir Ali Mughal**
- GitHub: [@MuntazirAliM](https://github.com/MuntazirAliM)
- LinkedIn: [linkedin.com/in/muntazir-ali-mughal](https://linkedin.com/in/muntazir-ali-mughal)

---

*Built with Python 3.11 · Groq API · March 2026*