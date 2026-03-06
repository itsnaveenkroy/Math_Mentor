# Math Mentor — Complete Code Explanation

A step-by-step walkthrough of the entire project: what each file does, when it runs, and how all the pieces connect.

---

## Table of Contents

1. [Big Picture — What Is This Project?](#1-big-picture--what-is-this-project)
2. [Project Structure — File Map](#2-project-structure--file-map)
3. [Technologies Used](#3-technologies-used)
4. [How the App Starts](#4-how-the-app-starts)
5. [Step-by-Step: What Happens When a User Submits a Problem](#5-step-by-step-what-happens-when-a-user-submits-a-problem)
   - [Phase 1: Input Collection](#phase-1-input-collection-apppy)
   - [Phase 2: The Orchestrator Takes Over](#phase-2-the-orchestrator-takes-over-agentsorchestratepy)
   - [Phase 3: Parser Agent](#phase-3-parser-agent-agentsparser_agentpy)
   - [Phase 4: Router Agent + RAG Retrieval](#phase-4-router-agent--rag-retrieval)
   - [Phase 5: Solver Agent + SymPy Computation](#phase-5-solver-agent--sympy-computation)
   - [Phase 6: Verifier Agent](#phase-6-verifier-agent-agentsverifier_agentpy)
   - [Phase 7: Explainer Agent](#phase-7-explainer-agent-agentsexplainer_agentpy)
   - [Phase 8: Memory Storage](#phase-8-memory-storage)
   - [Phase 9: Results Display + HITL](#phase-9-results-display--hitl)
6. [Input Handlers Deep Dive](#6-input-handlers-deep-dive)
   - [Text Handler](#text-handler-input_handlerstext_handlerpy)
   - [Image Handler (OCR)](#image-handler-ocr-input_handlersimage_handlerpy)
   - [Audio Handler (Speech-to-Text)](#audio-handler-speech-to-text-input_handlersaudio_handlerpy)
7. [RAG System Deep Dive](#7-rag-system-deep-dive)
   - [Knowledge Base Indexing](#knowledge-base-indexing-ragknowledge_basepy)
   - [Embeddings](#embeddings-ragembeddingspy)
   - [Retriever](#retriever-ragretrieverpy)
8. [Memory System Deep Dive](#8-memory-system-deep-dive)
9. [LLM Layer Deep Dive](#9-llm-layer-deep-dive)
10. [Math Tools Deep Dive](#10-math-tools-deep-dive)
11. [Configuration](#11-configuration)
12. [Data Flow Diagram](#12-data-flow-diagram)
13. [Key Design Decisions](#13-key-design-decisions)

---

## 1. Big Picture — What Is This Project?

Math Mentor is a **multimodal AI math tutor** that accepts math problems via **text, image (OCR), or audio (speech-to-text)**, then solves them using a **5-agent pipeline** with step-by-step explanations.

The core idea: instead of one monolithic LLM call, we break the solving process into 5 specialized agents that each handle one job. Each agent is an LLM call with a focused system prompt. This produces more reliable, verifiable, and explainable results than a single "solve this" prompt.

The pipeline:

```
User Input → [Parse] → [Route] → [Solve] → [Verify] → [Explain] → Display
```

---

## 2. Project Structure — File Map

```
math_mentor/
│
├── app.py                          # Streamlit UI (entry point — you run this)
│
├── agents/                         # The 5-agent pipeline
│   ├── orchestrator.py             # Coordinates all 5 agents in sequence
│   ├── parser_agent.py             # Agent 1: cleans and structures the raw input
│   ├── router_agent.py             # Agent 2: decides strategy + fetches KB context
│   ├── solver_agent.py             # Agent 3: actually solves the math
│   ├── verifier_agent.py           # Agent 4: double-checks the answer
│   └── explainer_agent.py          # Agent 5: writes a student-friendly explanation
│
├── input_handlers/                 # Multimodal input processing
│   ├── text_handler.py             # Basic text validation
│   ├── image_handler.py            # EasyOCR + math symbol cleanup
│   └── audio_handler.py            # Speech-to-text + WAV normalization
│
├── rag/                            # Retrieval Augmented Generation
│   ├── knowledge_base.py           # Loads + chunks + indexes markdown docs
│   ├── embeddings.py               # Sentence-transformer embedding wrapper
│   └── retriever.py                # Queries ChromaDB for relevant context
│
├── knowledge_base/                 # Curated math reference docs (20 markdown files)
│   ├── algebra_quadratic.md        # Quadratic equations formulas + methods
│   ├── calculus_derivatives.md     # Derivative rules + examples
│   ├── common_mistakes.md          # Common math pitfalls
│   └── ...                         # (20 total covering algebra, calc, prob, linalg, trig)
│
├── memory/                         # Persistent memory system
│   └── memory_store.py             # ChromaDB for problem history + JSON for stats
│
├── utils/                          # Shared utilities
│   ├── config.py                   # Env var loading (.env file)
│   ├── llm.py                      # LLM calls (Groq primary, Ollama fallback)
│   └── math_tools.py               # SymPy computation helpers
│
├── data/                           # Persisted data (auto-generated)
│   ├── chroma_db/                  # Knowledge base vector store
│   └── memory/                     # Problem memory vector store + stats JSON
│
├── pyproject.toml                  # Python package config + dependencies
├── requirements.txt                # Frozen dependency versions
├── packages.txt                    # System-level dependencies (for deployment)
└── README.md                       # Project documentation
```

---

## 3. Technologies Used

| Technology | What It Is | Where It's Used | Why |
|---|---|---|---|
| **Python 3.12** | Programming language | Everywhere | Main implementation language |
| **Streamlit** | Web UI framework | `app.py` | Turns Python scripts into web apps with minimal code. No HTML/CSS needed |
| **Groq API** | LLM inference | `utils/llm.py` | Fast inference on open models (default: `openai/gpt-oss-120b`). Uses OpenAI-compatible protocol |
| **Ollama** | Local LLM runner | `utils/llm.py` | Fallback when Groq is down. Runs models locally (default: `llama3.2`) |
| **OpenAI Python SDK** | HTTP client | `utils/llm.py` | Talks to both Groq and Ollama (they all use OpenAI's chat completions protocol) |
| **ChromaDB** | Vector database | `rag/`, `memory/` | Stores embeddings for similarity search. Used for both the knowledge base (RAG) and problem memory |
| **all-MiniLM-L6-v2** | Embedding model | `rag/embeddings.py` | Converts text to 384-dim vectors for similarity search. Runs locally via ONNX (no API needed) |
| **SymPy** | Symbolic math library | `utils/math_tools.py` | Solves equations, derivatives, integrals, limits, matrices algebraically. Gives ground-truth to verify LLM answers |
| **EasyOCR** | Optical Character Recognition | `input_handlers/image_handler.py` | Reads text from uploaded images. Uses deep learning (CRAFT text detector + recognition CNN) |
| **SpeechRecognition** | Audio transcription wrapper | `input_handlers/audio_handler.py` | Converts speech audio to text using Google's free Speech API |
| **pydub + ffmpeg** | Audio format conversion | `input_handlers/audio_handler.py` | Converts browser-recorded WebM/OGG/float32-WAV to standard 16-bit PCM WAV |
| **Pillow (PIL)** | Image processing | `input_handlers/image_handler.py` | Opens and converts uploaded images to numpy arrays for EasyOCR |
| **NumPy** | Numerical computing | `input_handlers/image_handler.py` | Converts PIL images to arrays; general numerical operations |
| **python-dotenv** | Environment config | `utils/config.py` | Loads API keys and settings from `.env` file |

---

## 4. How the App Starts

When you run `streamlit run app.py`, here's what happens:

### Step 1: Streamlit boots up
Streamlit starts a local web server (default `http://localhost:8501`). It imports and executes `app.py`.

### Step 2: Imports load the modules
```python
from agents.orchestrator import run_pipeline       # the main pipeline function
from input_handlers.text_handler import process_text_input
from input_handlers.image_handler import extract_text_from_image
from input_handlers.audio_handler import transcribe_audio
from memory.memory_store import store_feedback, get_topic_stats, store_correction
```
These imports trigger loading of sub-modules. Notably:
- `utils/config.py` reads your `.env` file (API keys, model names, paths)
- EasyOCR and ChromaDB are **lazily initialized** (not loaded until first use, to keep startup fast)

### Step 3: Page renders
Streamlit calls `main()` which renders the UI:
- Sidebar with input mode radio buttons and topic history
- Main area with input widgets (text box, file uploader, or audio recorder)
- The "Solve" button

### Step 4: Streamlit's execution model
**Important:** Streamlit re-runs the ENTIRE `app.py` script on every user interaction (button click, text input, file upload). This is Streamlit's core model — every widget interaction triggers a full top-to-bottom script re-execution. Session state (`st.session_state`) persists across re-runs.

---

## 5. Step-by-Step: What Happens When a User Submits a Problem

Let's trace exactly what happens when a user types "Solve x^2 - 5x + 6 = 0" and clicks **Solve**.

---

### Phase 1: Input Collection (`app.py`)

**What happens:** The user's text is captured from the Streamlit text area.

```python
raw_text = st.text_area("Enter your math problem:", ...)
```

For **image input**, the flow is:
1. User uploads image → `st.file_uploader` gives us bytes
2. `extract_text_from_image(uploaded.getvalue())` runs EasyOCR → returns extracted text
3. Text is shown to user for review/editing (human-in-the-loop)

For **audio input**, the flow is:
1. User uploads file or records via microphone
2. `transcribe_audio(audio_bytes, filename)` converts speech → text
3. Text is shown to user for review/editing (human-in-the-loop)

**Human-in-the-loop (HITL):** If the input came from OCR or audio, the extracted text is shown in an editable text area. The user can fix mistakes before solving. If they do edit, the correction is saved to memory via `store_correction()` for future self-learning.

When the user clicks **Solve**, this fires:

```python
if st.button("Solve", type="primary", disabled=not raw_text.strip()):
    _run_and_display(raw_text, input_source)
```

---

### Phase 2: The Orchestrator Takes Over (`agents/orchestrator.py`)

**What happens:** `run_pipeline(raw_text, input_source)` is called. This is the central coordinator that runs all 5 agents in sequence.

```python
def run_pipeline(raw_text, input_source="text"):
    trace = []          # records what each agent did (for transparency)
    start_time = time.time()
    result = { ... }    # big dict that accumulates all output
```

The orchestrator:
1. Calls each agent in order: Parse → Route → Solve → Verify → Explain
2. Times each agent call
3. Records a "trace" of inputs/outputs for each agent
4. Handles errors gracefully (if one agent fails, it still returns what it has)
5. Assembles the final result dict that the UI displays

---

### Phase 3: Parser Agent (`agents/parser_agent.py`)

**What happens:** Takes raw, potentially messy text and produces a structured JSON representation.

**When:** First agent to run. Always runs.

**How it works:**
1. Builds a system prompt telling the LLM it's a "math problem parser"
2. Adds context about the input source (e.g., "this came from OCR, watch for garbled symbols")
3. Calls the LLM via `chat(messages, want_json=True)`
4. The LLM returns structured JSON with fields like:

```json
{
    "problem_text": "Solve x^2 - 5x + 6 = 0",
    "problem": "Solve x^2 - 5x + 6 = 0",
    "topic": "algebra",
    "variables": ["x"],
    "constraints": [],
    "given": ["x^2 - 5x + 6 = 0"],
    "find": "values of x",
    "needs_clarification": false
}
```

**Key fields:**
- `problem`: cleaned-up problem statement (used by all downstream agents)
- `topic`: first guess at the math category
- `variables`: what variables are involved
- `given` / `find`: what's provided and what's asked for
- `needs_clarification`: if `true`, the orchestrator flags this for the user
- `constraints`: any conditions like "x > 0"

**Fallback:** If the LLM fails entirely, the parser returns the raw text as-is (so the pipeline doesn't crash).

---

### Phase 4: Router Agent + RAG Retrieval

**What happens:** Decides the solving strategy and pulls relevant reference material from the knowledge base.

**When:** Runs after the Parser. Uses the Parser's structured output.

**Two things happen simultaneously:**

#### 4a. RAG Context Retrieval (`rag/retriever.py`)

1. `retrieve_context(problem_text, k=5)` is called
2. This triggers `index_knowledge_base()` on first run — reads all 20 markdown files from `knowledge_base/`, chunks them into ~500-char pieces, and stores them in ChromaDB with embeddings
3. The problem text is embedded using **all-MiniLM-L6-v2** (384-dimensional vector)
4. ChromaDB performs **cosine similarity search** to find the top-5 most relevant chunks
5. Returns chunks with similarity scores

For "Solve x^2 - 5x + 6 = 0", it would pull chunks from `algebra_quadratic.md` containing factoring methods, the quadratic formula, etc.

#### 4b. Routing Decision (`agents/router_agent.py`)

1. Sends the parsed problem to the LLM with a routing-focused system prompt
2. LLM returns:

```json
{
    "topic": "algebra",
    "strategy": "Factor the quadratic and solve for roots",
    "tools_needed": ["equation_solver"],
    "difficulty": "easy",
    "subtopics": ["quadratic", "factoring"]
}
```

**Key output:**
- `tools_needed`: tells the Solver which SymPy tools to try (equation_solver, derivative, integral, limit, matrix_op, probability)
- `strategy`: a human-readable description of the approach
- `context`: the RAG chunks (attached by the Router)

---

### Phase 5: Solver Agent + SymPy Computation

**What happens:** Actually solves the math problem using LLM reasoning + computational verification.

**When:** Runs after the Router. Has access to the strategy, tools list, KB context, and similar past problems.

**Two-pronged approach:**

#### 5a. SymPy Computation First (`utils/math_tools.py`)

Before calling the LLM, the solver tries to compute the answer using SymPy:

```python
comp_result = _try_computation(problem_text, tools_needed, parsed_problem)
```

For our example, since `tools_needed` includes `equation_solver`:
1. It calls `math_tools.solve_equation("x^2 - 5x + 6 = 0")`
2. SymPy parses the equation, calls `sp.solve()`, returns `[2, 3]`
3. This gives us **mathematical ground truth** to compare against the LLM

The `math_tools.py` module provides:
- `solve_equation()` — algebraic equation solving
- `compute_derivative()` — symbolic differentiation
- `compute_integral()` — symbolic integration
- `compute_limit()` — limit computation
- `compute_matrix_op()` — determinant, inverse, eigenvalues
- `compute_probability()` — nCr/nPr style computations
- `safe_eval()` — safe expression evaluation

The SymPy result is used as `"Computational verification"` context for the LLM.

#### 5b. LLM Solving

The solver builds a rich prompt with:
- The problem text
- The Router's strategy hint
- KB context (relevant formulas and methods from the knowledge base)
- SymPy's computational result (if available)
- Similar past problems from memory (if any)

The LLM returns step-by-step solution:

```json
{
    "solution": "x = 2, x = 3",
    "steps": [
        {"step": 1, "description": "Write the equation", "work": "x^2 - 5x + 6 = 0"},
        {"step": 2, "description": "Factor", "work": "(x-2)(x-3) = 0"},
        {"step": 3, "description": "Set each factor to zero", "work": "x = 2 or x = 3"}
    ],
    "method": "factoring",
    "confidence": 0.95
}
```

**Fallback:** If the LLM fails but SymPy succeeded, the solver uses the SymPy result as the answer (with lower confidence).

---

### Phase 6: Verifier Agent (`agents/verifier_agent.py`)

**What happens:** A separate LLM call double-checks the solver's work.

**When:** Runs after the Solver. Receives both the parsed problem and the solver's output.

**How:**
1. Sends the problem, proposed solution, solution steps, and method to a fresh LLM call with a verification-focused system prompt
2. If we have a SymPy result, includes it as a cross-reference
3. The LLM independently checks the solution and returns:

```json
{
    "is_correct": true,
    "confidence": 0.95,
    "issues": [],
    "suggestions": [],
    "verified_solution": "x = 2, x = 3"
}
```

**Why this matters:** The Verifier is a different LLM call with a different system prompt than the Solver. This means it approaches the problem fresh. If the Solver made an error, the Verifier has a good chance of catching it. This is the "two heads are better than one" principle.

**If SymPy disagrees with the LLM:** The Verifier adds this to `issues` (e.g., "Sympy computed: [2, 3]").

---

### Phase 7: Explainer Agent (`agents/explainer_agent.py`)

**What happens:** Generates a student-friendly explanation of the solution.

**When:** Last agent to run. Has access to everything: parsed problem, solution, verification result.

**Special behavior:** It checks the user's topic history from memory:
```python
stats = get_topic_stats()
if stats[topic]["avg_confidence"] < 0.6:
    # "this student has struggled with algebra before, be extra clear"
```

If the user has low confidence scores on this topic, the Explainer is told to be more thorough.

Returns:
```json
{
    "explanation": "Step-by-step walk-through in plain language...",
    "key_concepts": ["factoring", "zero product property"],
    "common_mistakes": ["forgetting to check both roots"],
    "difficulty_rating": "easy",
    "related_topics": ["completing the square", "quadratic formula"]
}
```

---

### Phase 8: Memory Storage

**What happens:** The solved problem is stored in memory for future reference.

**When:** After all 5 agents complete, before returning results to the UI.

```python
store_problem(
    problem_text,
    result["answer"],
    topic=route_info.get("topic", "general"),
    confidence=result["confidence"],
    context_used=route_info.get("context", []),     # which KB chunks were used
    verifier_outcome=verification,                   # the verifier's assessment
)
```

This stores:
1. **In ChromaDB**: The problem text + solution as a document with semantic embedding (for similarity search later)
2. **In JSON file**: Topic statistics (count, average confidence, feedback counts)

Next time the user asks a similar problem, `find_similar_problems()` will find this one and feed it to the Solver as context.

---

### Phase 9: Results Display + HITL

**What happens:** The orchestrator returns the assembled result dict to `app.py`, which displays everything.

**UI sections displayed:**

1. **Answer** — The verified solution + confidence score with color-coded label
2. **Clarification warning** — If the Parser flagged ambiguity (`needs_clarification: true`)
3. **Re-check button** — If confidence < 60%, user can request a re-solve
4. **Method** — Which mathematical method was used
5. **Explanation** — Student-friendly walk-through
6. **Solution steps** — Expandable step-by-step work (from Solver)
7. **Verification notes** — Any issues the Verifier found
8. **Concepts and tips** — Key concepts, common mistakes, related topics (from Explainer)
9. **Similar past problems** — Problems the user asked before that are similar
10. **Knowledge base context** — Which KB chunks were used (with similarity scores)
11. **Agent pipeline trace** — Expandable view of what each agent received/returned and how long it took
12. **Feedback section** — "Was this helpful?" buttons + comment box

**Feedback loop:** When the user clicks "Yes, correct" / "Wrong answer" / "Confusing explanation", this is stored via `store_feedback()` and influences future topic stats.

---

## 6. Input Handlers Deep Dive

### Text Handler (`input_handlers/text_handler.py`)

The simplest handler. Does two things:
1. **Cleans whitespace** — Collapses multiple spaces, strips leading/trailing space
2. **Sanity check** — Uses regex to detect if the input contains math-like characters (`+`, `-`, `=`, `^`, digits, etc.) or math-like words ("solve", "derivative", "integral", etc.)

If the input is very short and doesn't look like math, it flags `needs_hitl: true`.

### Image Handler (OCR) (`input_handlers/image_handler.py`)

**Technology: EasyOCR**

EasyOCR is a deep learning OCR library that uses:
- **CRAFT** text detector (finds where text is in the image)
- **Recognition CNN** (reads the detected text regions)

It runs on CPU (GPU disabled: `gpu=False`) and supports English.

**The OCR pipeline:**

1. **Image loading**: `PIL.Image.open()` converts uploaded bytes to image, then `np.array()` for EasyOCR
2. **Text detection + recognition**: `reader.readtext(img_np)` returns list of `(bounding_box, text, confidence)` tuples
3. **Concatenation**: All detected text pieces are joined with spaces
4. **Math-specific cleanup** (`_clean_ocr_math()`): Regex patterns fix common OCR mistakes:
   - `V(` → `sqrt(` (OCR misreads √ as V)
   - `~` → `-` (tilde misread as minus)
   - `TT` → `pi` (OCR misreads π)
   - Various other symbol fixes
5. **Learned corrections** (`_apply_learned_corrections()`): Checks `corrections_log.json` for past user edits. If the user previously corrected "x2" to "x^2", future OCR outputs get the same correction applied automatically.
6. **Returns** text + confidence + `needs_hitl: true` (OCR always needs human review for math)

### Audio Handler (Speech-to-Text) (`input_handlers/audio_handler.py`)

This is the most complex handler because of browser audio format issues.

**The audio pipeline:**

1. **Format detection** (`_detect_audio_format()`): Inspects the first few bytes ("magic bytes") to determine the actual audio format — WAV, WebM, OGG, MP3, etc. This is important because browsers may produce different formats than what the filename suggests.

2. **WAV normalization** (`_normalize_wav_bytes()`): If it's a WAV file, the raw bytes are parsed manually (not using Python's `wave` module, which can't handle float32 WAV). This handles:
   - **Float32 WAV** (format tag 3) → convert to 16-bit PCM (format tag 1)
   - **Int32, Int24, 8-bit** → convert to 16-bit
   - **Stereo** → mix down to mono

3. **Resampling**: After normalization, uses **pydub + ffmpeg** to resample to 16kHz (browsers record at 48kHz, which can cause issues with speech APIs).

4. **For non-WAV** (WebM, OGG, etc.): Uses **pydub + ffmpeg** to convert to 16-bit mono WAV at 16kHz.

5. **SpeechRecognition**: The normalized WAV is loaded as `AudioData`, then `recognizer.recognize_google(audio)` sends it to Google's free Speech-to-Text API.

6. **Math phrase normalization** (`_normalize_math_phrases()`): Converts spoken math to symbols:
   - "square root of" → `sqrt`
   - "squared" → `^2`
   - "times" → `*`
   - "derivative of" → `diff`
   - "sine" → `sin`
   - etc.

7. **Learned corrections** applied (same as OCR — checks past user edits)

---

## 7. RAG System Deep Dive

RAG = **Retrieval Augmented Generation**. The idea: instead of relying only on what the LLM knows, we give it relevant reference material from a curated knowledge base.

### Knowledge Base Indexing (`rag/knowledge_base.py`)

**When it runs:** Once, on the first query. After that, Chrome DB already has the data.

**The 20 markdown files** in `knowledge_base/` cover:
- Algebra: quadratics, polynomials, systems, logarithms, sequences, inequalities
- Calculus: derivatives, integrals, limits, optimization
- Linear algebra: matrices, vectors, determinants
- Probability: basics, combinatorics, distributions
- Trigonometry: formulas
- Meta: common mistakes, solution templates, domain constraints

**Indexing process:**

1. Read each `.md` file
2. **Chunk it**: Split on `##` headers first. If a section is still > 500 chars, split on word boundaries with 50-char overlap. This ensures each chunk is a coherent piece of information.
3. **Generate deterministic IDs**: `MD5(filename + chunk_index)` so re-indexing doesn't create duplicates
4. **Store in ChromaDB**: Each chunk gets an embedding (all-MiniLM-L6-v2 generates a 384-dim vector) and is stored with metadata (source file, topic, chunk index)

### Embeddings (`rag/embeddings.py`)

Uses ChromaDB's **DefaultEmbeddingFunction** which runs **all-MiniLM-L6-v2** via ONNX Runtime.

- **Model**: all-MiniLM-L6-v2 (22M parameters, very fast)
- **Output**: 384-dimensional float vector
- **Runs locally**: No API call needed — the model runs on your CPU
- **Singleton pattern**: `_embed_fn` is created once and reused

### Retriever (`rag/retriever.py`)

**What it does:** Given a problem text, find the most relevant KB chunks.

```python
results = collection.query(
    query_texts=[query],     # ChromaDB auto-embeds this
    n_results=5,
    include=["documents", "metadatas", "distances"]
)
```

ChromaDB converts the query to a vector, then finds the 5 nearest vectors in the collection using **cosine distance** (configured via `"hnsw:space": "cosine"`).

**HNSW** (Hierarchical Navigable Small World) is the approximate nearest neighbor algorithm ChromaDB uses — it's very fast even with large collections.

Distance is converted to similarity: `similarity = 1.0 - (distance / 2.0)` (cosine distance ranges 0–2, so similarity ranges 0–1).

---

## 8. Memory System Deep Dive

The memory system (`memory/memory_store.py`) serves three purposes:

### 1. Problem History (ChromaDB)

Every solved problem is stored as a document in a ChromaDB collection called `problem_memory`:

```
Document: "Problem: Solve x^2 - 5x + 6 = 0\n\nSolution: x = 2, x = 3\n\nSources: algebra_quadratic.md"
Metadata: {topic: "algebra", confidence: 0.95, timestamp: 1709712345.6, verifier_correct: "True"}
```

When a new problem comes in, `find_similar_problems()` searches this collection. If it finds a match with > 60% similarity, those past solutions are passed to the Solver as context ("here's how a similar problem was solved before").

### 2. Topic Statistics (JSON file: `topic_stats.json`)

Tracks per-topic aggregated data:
```json
{
    "algebra": {"count": 15, "total_confidence": 13.5, "feedback_counts": {"helpful": 10, "wrong": 2}},
    "calculus": {"count": 8, "total_confidence": 5.6, "feedback_counts": {"confusing": 3}}
}
```

Used by:
- **Sidebar**: Shows the user their history ("Algebra: 15 problems, avg confidence 90%")
- **Explainer Agent**: If avg confidence is low for a topic, adds instructions to explain more thoroughly

### 3. Correction Memory (JSON file: `corrections_log.json`)

Stores user edits to OCR/audio transcriptions:
```json
[
    {"original": "x2 + 3x", "corrected": "x^2 + 3x", "source": "image", "timestamp": 1709712345.6}
]
```

These corrections are applied to future OCR/audio outputs automatically — the system learns from its mistakes.

### 4. User Feedback (JSON file: `feedback_log.json`)

Stores explicit feedback ("helpful", "wrong", "confusing", free-text comments). Used to update topic stats.

---

## 9. LLM Layer Deep Dive

`utils/llm.py` is the single point of contact for all LLM calls. Every agent calls `chat(messages, want_json=True)`.

### Dual-provider setup

```
User request
    ↓
Try Groq (cloud, default: openai/gpt-oss-120b)
    ↓ fails?
Try Ollama (local, default: llama3.2)
    ↓ fails?
Return None (agent handles gracefully)
```

Both providers use the **OpenAI chat completions protocol**, so we use the `openai` Python SDK to talk to both. Only the `base_url` and `api_key` differ.

### JSON extraction

Since all agents expect JSON, the `_extract_json()` function handles messy LLM output:
1. Try direct `json.loads()`
2. Strip markdown fences (` ```json ... ``` `)
3. Find the first `{...}` block with regex
4. If the response was truncated (common with free models hitting token limits), `_repair_truncated_json()` closes open strings/arrays/braces

### Token management

`max_tokens=1024` keeps responses short to avoid truncation on free models.

The system prompt gets a nudge appended: "Reply ONLY with valid JSON. Keep strings short."

---

## 10. Math Tools Deep Dive

`utils/math_tools.py` provides SymPy-based computation — this is the "ground truth" engine.

### Why both LLM and SymPy?

- **LLM**: Good at understanding word problems, choosing methods, generating explanations. Bad at precise arithmetic.
- **SymPy**: Does exact symbolic math. Can't understand "a farmer has 3 fields..." but will solve `3x + 7 = 22` perfectly.

Using both means the LLM reasons about the approach while SymPy verifies the computation.

### Key functions

| Function | Input | Output | Used For |
|---|---|---|---|
| `solve_equation(eq_str)` | `"x^2-5x+6=0"` | `{solutions: ["2", "3"]}` | Algebra |
| `compute_derivative(expr)` | `"x^3 + 2*x"` | `{derivative: "3*x^2 + 2"}` | Calculus |
| `compute_integral(expr)` | `"x^2"` | `{integral: "x^3/3 + C"}` | Calculus |
| `compute_limit(expr, var, pt)` | `"sin(x)/x", "x", "0"` | `{limit: "1"}` | Calculus |
| `compute_matrix_op(op, data)` | `"determinant", [[1,2],[3,4]]` | `{result: "-2"}` | Linear Algebra |
| `compute_probability(expr)` | `"10C3"` | `{result: "120"}` | Probability |
| `safe_eval(expr)` | `"sqrt(144)"` | `{numerical: 12.0}` | General |

### Expression parsing

SymPy's parser is configured with special transformations:
- `implicit_multiplication_application`: `2x` → `2*x`
- `convert_xor`: `x^2` → `x**2` (Python uses `**` for exponentiation, not `^`)

---

## 11. Configuration

All configuration lives in `utils/config.py` which reads from a `.env` file:

```bash
# .env file
GROQ_API_KEY=gsk_...                 # Required: your Groq API key
GROQ_MODEL=openai/gpt-oss-120b                        # Default model
OLLAMA_BASE_URL=http://localhost:11434/v1              # Local Ollama endpoint
OLLAMA_MODEL=llama3.2                                  # Local model for fallback
CHROMA_PERSIST_DIR=./data/chroma_db                    # Where KB vectors are stored
MEMORY_PERSIST_DIR=./data/memory                       # Where problem history is stored
MAX_RETRIEVAL_K=5                                      # How many KB chunks to retrieve
CONFIDENCE_THRESHOLD=0.7                               # Below this → flag for human review
```

---

## 12. Data Flow Diagram

```
┌─────────────┐
│  User Input  │
│ Text/Image/  │
│    Audio     │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Input Handler │  ── image_handler.py (OCR + math cleanup)
│              │  ── audio_handler.py (STT + WAV normalize)
│              │  ── text_handler.py  (validation)
└──────┬───────┘
       │ cleaned text
       ▼
┌──────────────┐
│  HITL Edit   │  User reviews/fixes extracted text
│  (app.py)    │  Corrections saved to memory
└──────┬───────┘
       │ final text
       ▼
┌──────────────────────────────────────────────────────────────────┐
│                     ORCHESTRATOR (orchestrator.py)                │
│                                                                  │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │ 1.PARSER │───▶│ 2.ROUTER │───▶│ 3.SOLVER │───▶│4.VERIFIER│  │
│  │          │    │          │    │          │    │          │  │
│  │ LLM call │    │ LLM call │    │ LLM call │    │ LLM call │  │
│  │ Structures│   │ + RAG    │    │ + SymPy  │    │ Cross-   │  │
│  │ the input│    │ retrieval│    │ compute  │    │ checks   │  │
│  └──────────┘    └────┬─────┘    └────┬─────┘    └──────────┘  │
│                       │               │                │        │
│                  ┌────▼─────┐    ┌────▼─────┐    ┌────▼────┐   │
│                  │ChromaDB  │    │ SymPy    │    │5.EXPLAIN│   │
│                  │(KB search)│   │(math_tools│   │ LLM call│   │
│                  └──────────┘    └──────────┘    └─────────┘   │
│                                                                  │
│  Also: find_similar_problems() from Memory at start              │
│  Also: store_problem() to Memory at end                         │
└──────────────────────────┬───────────────────────────────────────┘
                           │ result dict
                           ▼
                   ┌───────────────┐
                   │  UI Display   │
                   │   (app.py)    │
                   │               │
                   │ • Answer      │
                   │ • Explanation │
                   │ • Steps       │
                   │ • Verification│
                   │ • Concepts    │
                   │ • Agent trace │
                   │ • Feedback    │
                   └───────────────┘
```

---

## 13. Key Design Decisions

### Why 5 separate agents instead of one big prompt?

**Separation of concerns.** Each agent has a focused job with a short system prompt. This means:
- Each prompt is optimized for its specific task
- If one step fails, others can still work
- The Verifier is independent from the Solver (catches more errors)
- Each step is traceable and debuggable (shown in the agent trace UI)

### Why both Groq and Ollama?

**Reliability.** Cloud LLM APIs can have downtime or rate limits. Having a local Ollama fallback means the app works even when the cloud is down. Both speak the same OpenAI protocol, so the code is almost identical.

### Why SymPy alongside the LLM?

**Trust but verify.** LLMs are great at reasoning but can make arithmetic errors. SymPy gives exact mathematical results. The Solver uses SymPy results as "computational verification" context, and the Verifier can cross-check against SymPy's answer.

### Why ChromaDB for both RAG and Memory?

**Semantic search.** ChromaDB lets us find "similar" content using embedding vectors. For the knowledge base, "similar" means "relevant formulas and methods." For memory, "similar" means "the user asked something like this before." Both need the same capability.

### Why manual WAV byte parsing instead of Python's wave module?

**Browser compatibility.** Python's `wave` module only supports PCM WAV (format tag 1). Browsers produce float32 WAV (format tag 3) from `MediaRecorder`. Our `_normalize_wav_bytes()` function parses the raw RIFF/WAV header bytes using `struct.unpack()` and handles the conversion manually.

### Why human-in-the-loop (HITL) for OCR and audio?

**Math notation is hard.** OCR regularly misreads `√` as `V`, and speech-to-text doesn't know that "x squared" means `x^2`. Showing the user the extracted text and letting them fix it before solving prevents the entire pipeline from working on garbage input.

### Why save user corrections?

**Self-learning loop.** If a user corrects "x2" to "x^2" from OCR output, that correction is stored. Next time OCR produces "x2", the system automatically applies the learned correction. This gets better over time with usage.

---

*This document covers every file and every data flow in the project. Each technology choice, each function call, each agent interaction is explained above. Use it to understand the codebase or to explain it to evaluators.*
