# 🧠 RAG System — Retrieval-Augmented Generation Pipeline

A production-grade Retrieval-Augmented Generation (RAG) system built with **LangChain**, **Groq (LLaMA 3.3 70B)**, **ChromaDB**, and **HuggingFace Embeddings**. Ask complex questions across multiple documents and get cited, synthesised answers — not just keyword matches.

---

## 📋 Table of Contents

- [What is RAG?](#-what-is-rag)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Usage](#-usage)
- [CLI Commands](#-cli-commands)
- [Use Cases](#-use-cases)
- [How It Works](#-how-it-works)
- [Retrieval Strategies](#-retrieval-strategies)
- [Troubleshooting](#-troubleshooting)
- [Future Scope](#-future-scope)

---

## 🤔 What is RAG?

**Retrieval-Augmented Generation** is a technique that extends a Large Language Model (LLM) with your own private documents. Instead of relying solely on the model's training data, RAG:

1. **Retrieves** the most relevant chunks from your documents using semantic search
2. **Augments** the LLM prompt with that retrieved context
3. **Generates** an answer grounded in your actual documents — not hallucinated

This means the LLM can answer questions about documents it has never seen during training, with citations you can verify.

```
User Question
     │
     ▼
Embedding Model  ──►  Semantic Search  ──►  ChromaDB
                                                │
                                                ▼ Relevant Chunks
                                         LLM (Groq / LLaMA)
                                                │
                                                ▼
                                     Cited, Grounded Answer
```

---

## ✨ Features

| Feature | Description |
|---|---|
| **Multi-format ingestion** | Supports PDF, TXT, and DOCX documents |
| **High-quality embeddings** | `all-mpnet-base-v2` (768-dim) for strong semantic understanding |
| **MMR Retrieval** | Maximal Marginal Relevance — diverse, non-duplicate chunk selection |
| **Multi-Query Retrieval** | LLM rewrites your question into multiple variants for broader coverage |
| **Source citations** | Every answer shows which files and pages it drew from |
| **Strategy switching** | Toggle between retrieval strategies at runtime without restarting |
| **Debug tools** | Inspect exactly what chunks are retrieved for any query |
| **Strategy comparison** | Side-by-side view of what MMR vs MultiQuery fetches |
| **Persistent vectorstore** | ChromaDB persisted to disk — ingest once, query forever |
| **Graceful error handling** | Dimension mismatch auto-resolved on re-ingest |

---

## 🛠 Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **LLM** | Groq — LLaMA 3.3 70B Versatile | Answer generation |
| **Embeddings** | HuggingFace — all-mpnet-base-v2 | Semantic vector encoding |
| **Vectorstore** | ChromaDB | Persistent local vector database |
| **Framework** | LangChain + LCEL | Chain orchestration |
| **Document loaders** | PyPDF, Docx2txt, TextLoader | Multi-format ingestion |
| **Retrieval** | MMR + MultiQueryRetriever | Advanced retrieval strategies |
| **Config** | python-dotenv | API key management |

---

## 📁 Project Structure

```
rag-project/
│
├── data/                        # ← Put your documents here (PDF, TXT, DOCX)
│
├── chroma_db/                   # Auto-created on first ingest (vector database)
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py           # Loads documents from data/
│   ├── text_splitter.py         # Splits docs into overlapping chunks
│   ├── embedding.py             # HuggingFace embedding model factory
│   ├── vectorstore.py           # ChromaDB create/persist logic
│   ├── retriever.py             # MMR + MultiQuery retriever factories
│   ├── llm.py                   # Groq LLM factory
│   ├── rag_chain.py             # LCEL RAG chain (simple + with-sources variants)
│   └── debug.py                 # Retrieval diagnostics toolkit
│
├── main.py                      # CLI entry point
├── requirements.txt
├── .env                         # API keys (never commit this)
└── README.md
```

---

## 🏗 Architecture

```
                        ┌─────────────────────────────────────────┐
                        │              INGESTION PIPELINE          │
                        │                                          │
  data/ (PDF/TXT/DOCX)  │                                          │
         │              │  data_loader → text_splitter →           │
         └──────────────►  embedding model → ChromaDB              │
                        │                                          │
                        └─────────────────────────────────────────┘

                        ┌─────────────────────────────────────────┐
                        │               QUERY PIPELINE             │
                        │                                          │
  User Question         │  MultiQueryRetriever                     │
         │              │    │ generates 3 query variants           │
         └──────────────►    │ each hits MMR retriever             │
                        │    │ results deduplicated                 │
                        │    ▼                                      │
                        │  format_docs_with_sources()              │
                        │    │ numbered [Source N] context blocks   │
                        │    ▼                                      │
                        │  ChatPromptTemplate (System + Human)     │
                        │    │                                      │
                        │    ▼                                      │
                        │  Groq LLM (LLaMA 3.3 70B)               │
                        │    │                                      │
                        │    ▼                                      │
                        │  Answer + Source Citations               │
                        └─────────────────────────────────────────┘
```

---

## ⚙️ Installation

### Prerequisites

- Python 3.9 or higher
- A free [Groq API key](https://console.groq.com) (takes 2 minutes)
- Optional: A [HuggingFace token](https://huggingface.co/settings/tokens) (removes rate-limit warnings)

### Step 1 — Clone or download the project

```bash
git clone https://github.com/your-username/rag-project.git
cd rag-project
```

### Step 2 — Create and activate a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Configure API keys

Create a `.env` file in the project root:

```env
GROQ_API_KEY=gsk_your_key_here
HUGGINGFACEHUB_API_TOKEN=hf_your_token_here    # optional but recommended
```

> ⚠️ Never commit `.env` to version control. Add it to `.gitignore`.

### Step 5 — Add your documents

Drop your PDF, TXT, or DOCX files into the `data/` folder:

```
data/
├── research_paper.pdf
├── notes.txt
└── report.docx
```

### Step 6 — Ingest and query

```bash
python main.py
# Choose option 1 to ingest documents (first run)
# Choose option 2 on subsequent runs to go straight to querying
```

The embedding model (`all-mpnet-base-v2`, ~438MB) downloads automatically on first run. This is a one-time download.

---

## 🔑 Configuration

All tunable parameters are constants at the top of each module — one place to change, no hunting through business logic.

| File | Constant | Default | Effect |
|---|---|---|---|
| `embedding.py` | `MODEL_NAME` | `all-mpnet-base-v2` | Embedding model |
| `retriever.py` | `DEFAULT_K` | `6` | Final chunks returned |
| `retriever.py` | `FETCH_K` | `20` | MMR candidate pool size |
| `retriever.py` | `LAMBDA_MULT` | `0.5` | MMR diversity (0=diverse, 1=relevant) |
| `retriever.py` | `N_QUERY_VARIANTS` | `3` | MultiQuery paraphrase count |
| `llm.py` | `TEMPERATURE` | `0.2` | LLM creativity (lower = more factual) |
| `llm.py` | `MAX_TOKENS` | `2048` | Max answer length |
| `text_splitter.py` | `chunk_size` | `800` | Characters per chunk |
| `text_splitter.py` | `chunk_overlap` | `150` | Overlap between chunks |

> **Changing the embedding model** requires re-ingesting all documents (option 1) since vector dimensions change.

---

## 🚀 Usage

### First run (ingest + query)

```bash
python main.py

# Prompt: Do you want to (1) Re-ingest documents or (2) Just query? → type 1
```

### Subsequent runs (query only)

```bash
python main.py

# Prompt: Do you want to (1) Re-ingest documents or (2) Just query? → type 2
```

### Example session

```
❓ Question: What problem does the transformer architecture solve compared to RNNs?

⏳ Thinking...

💬 Answer:
The transformer architecture solves the sequential computation bottleneck
of RNNs. As [Source 1] (1706.03762v7.pdf, page 2) explains, RNNs process
tokens one at a time, preventing parallelisation across the input sequence.
The transformer replaces recurrence entirely with self-attention, allowing
all positions to be processed simultaneously...

📚 Sources:
  • 1706.03762v7.pdf  (page 2)
  • 1706.03762v7.pdf  (page 5)

❓ Question: Compare the concepts explained across all documents.

⏳ Thinking...

💬 Answer:
Across the four documents, several interconnected themes emerge...
[Source 1] covers attention mechanisms and parallelisation in deep learning.
[Source 2] addresses geometric foundations...
```

---

## 🎮 CLI Commands

| Command | Description |
|---|---|
| `exit` / `quit` / `q` | Exit the program |
| `debug` | Inspect exactly which chunks are retrieved for a test query |
| `compare` | Run a query through both MMR and MultiQuery side-by-side |
| `mode` | Switch retriever strategy without restarting (`multi_query` / `mmr`) |
| `help` | Show the command menu |

### Debug example

```
❓ Question: debug
  Enter query to debug: What is self-attention?

──────────────────────────────────────────────────────────────────────
🔍 DEBUG  |  retriever=multi_query  |  k=6
Query: What is self-attention?
──────────────────────────────────────────────────────────────────────

📄 Retrieved 6 chunks

  ┌─ Chunk 1 ──────────────────────────────────────
  │  Source : 1706.03762v7.pdf
  │  Page   : 3
  │  Length : 742 chars
  │  Preview: An attention function can be described as mapping a query...
  └────────────────────────────────────────────────
```

---

## 💡 Use Cases

### 1. Research & Academic
Query large collections of research papers. Ask comparative questions across multiple papers, extract methodologies, find contradictions, or synthesise findings — without reading every page.

```
"What evaluation metrics are used across these papers?"
"Compare the experimental setups described in the documents."
```

### 2. Legal & Compliance
Analyse contracts, ISO standards, and regulatory documents. Ask precise questions and get answers with exact page citations for auditing.

```
"What are the mandatory requirements in section 4?"
"What clauses relate to liability and indemnification?"
```

### 3. Technical Documentation
Query engineering specs, API docs, and internal wikis. Ideal for onboarding engineers or answering questions across sprawling documentation sets.

```
"How does the authentication flow work?"
"What are the rate limits described in the docs?"
```

### 4. Business Intelligence
Analyse reports, meeting notes, and strategy documents. Extract key insights, compare quarters, or find action items across large document sets.

```
"What were the key risks identified across all quarterly reports?"
"Summarise the strategic priorities mentioned in these documents."
```

### 5. Education & Learning
Upload textbooks, lecture notes, or study materials. Ask concept questions, request explanations, or test your understanding.

```
"Explain backpropagation as described in the notes."
"What are the key differences between supervised and unsupervised learning?"
```

---

## 🔬 How It Works

### Ingestion Pipeline

```
1. data_loader.py      Recursively scans data/ for .pdf, .txt, .docx
                       Each file becomes a list of LangChain Documents

2. text_splitter.py    RecursiveCharacterTextSplitter splits each doc
                       into 800-char chunks with 150-char overlap
                       Separators tried in order: paragraph → line → sentence → word

3. embedding.py        all-mpnet-base-v2 encodes each chunk into a 768-dim vector
                       Vectors are L2-normalised for cosine similarity

4. vectorstore.py      ChromaDB stores (vector, text, metadata) tuples
                       Persisted to chroma_db/ on disk
```

### Query Pipeline

```
1. User question arrives at main.py

2. MultiQueryRetriever (retriever.py)
   The LLM generates 3 paraphrased versions of the question
   Each variant is run through the MMR retriever independently
   All results are merged and deduplicated

3. MMR Retriever (retriever.py)
   For each query variant:
   - Fetch 20 candidate chunks by cosine similarity
   - Re-rank to maximise both relevance AND diversity (λ=0.5)
   - Return top 6 diverse chunks

4. format_docs_with_sources() (rag_chain.py)
   Each chunk is prefixed: [Source N] (filename, page)
   Gives the LLM explicit citation handles

5. ChatPromptTemplate (rag_chain.py)
   System prompt instructs the LLM to:
   - Synthesise across all sources
   - Cite by [Source N]
   - Think step-by-step for complex queries
   - Use partial information rather than refusing

6. Groq LLM generates the answer

7. Answer + deduplicated source list returned to main.py
```

---

## 🔀 Retrieval Strategies

### MMR — Maximal Marginal Relevance
Best for: focused factual questions where you want the most relevant and non-redundant chunks.

```python
# Fetch 20 candidates, re-rank for diversity, return 6
search_type="mmr"
search_kwargs={"k": 6, "fetch_k": 20, "lambda_mult": 0.5}
```

### MultiQuery + MMR (default)
Best for: abstract, comparative, or broad questions where a single phrasing may not match any chunk.

```
"Compare concepts across documents"
    ↓ LLM generates:
    → "What are the main ideas in each document?"
    → "Identify themes common to all documents"
    → "Summarise the key concepts discussed"
    Each hits MMR → deduplicated union returned
```

Switch strategies at runtime:
```
❓ Question: mode
  Select mode: mmr
```

---

## 🔧 Troubleshooting

### `InvalidArgumentError: Collection expecting dimension 384, got 768`
You changed the embedding model but the old ChromaDB still exists.

```bash
# Delete the old database and re-ingest
Remove-Item -Recurse -Force chroma_db     # Windows PowerShell
rm -rf chroma_db                          # macOS / Linux

python main.py  # choose option 1
```

### `GROQ_API_KEY not found`
Your `.env` file is missing or the key is wrong.

```env
# .env
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx
```

### `No documents found in 'data'`
Your `data/` folder is empty or contains unsupported file types.
Supported formats: `.pdf`, `.txt`, `.docx`

### Answer says "I couldn't find sufficient information"
Run `debug` to see what's being retrieved:
```
❓ Question: debug
  Enter query to debug: your question here
```
If retrieved chunks are irrelevant, try `mode → mmr` or rephrase your question. If chunks look right but the answer is still poor, increase `MAX_TOKENS` in `llm.py`.

### HuggingFace rate limit warning
```
Warning: You are sending unauthenticated requests to the HF Hub.
```
This is harmless. To remove it, add your HF token to `.env`:
```env
HUGGINGFACEHUB_API_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
```

---

## 🔮 Future Scope

### Near-term enhancements
- **Conversational memory** — multi-turn Q&A with chat history via `ConversationBufferMemory`
- **Reranking** — add a cross-encoder reranker (e.g. `ms-marco-MiniLM`) after MMR for precision on technical queries
- **Streaming output** — token-by-token answer streaming for better UX on long responses
- **Score thresholding** — filter out chunks below a similarity score to reduce noise

### Medium-term
- **Web UI** — Streamlit or Gradio frontend with drag-and-drop document upload
- **Hybrid search** — combine dense (vector) + sparse (BM25) retrieval for better keyword precision
- **Metadata filtering** — query only specific files or date ranges using Chroma metadata filters
- **Evaluation harness** — RAGAS metrics (faithfulness, answer relevance, context recall) for systematic quality measurement

### Advanced
- **Agentic RAG** — LangGraph-based agent that decides when to retrieve, when to reason, when to ask for clarification
- **GraphRAG** — knowledge graph over document entities for multi-hop reasoning
- **Fine-tuned embeddings** — domain-adapted embedding model trained on your document corpus
- **Cloud deployment** — containerised with Docker, deployable to AWS/GCP/Azure with a REST API layer

---

## 📦 Requirements

```
langchain
langchain-core
langchain-community
langchain-text-splitters
langchain-huggingface
langchain-groq
langchain-chroma
pypdf
docx2txt
unstructured
python-dotenv
tqdm
sentence-transformers
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## 🙏 Credits

Built following the RAG fundamentals course by **Krish Naik** on YouTube, extended with production-grade patterns including MMR retrieval, MultiQueryRetriever, LCEL chain design, source attribution, and strategy-switching CLI.

---

*Built with LangChain · Groq · ChromaDB · HuggingFace*
