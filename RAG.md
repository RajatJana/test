# ClarifAI - RAG Implementation Design

## 1. Overview

ClarifAI currently generates user stories from scratch for every new set of requirements. Each generation is independent -- the system has no memory of what it produced before.

RAG (Retrieval-Augmented Generation) introduces a **self-improving feedback loop**:

1. **Generate** user stories from requirements (existing flow)
2. **Embed & Store** the generated stories in a vector database (new)
3. **Retrieve** semantically similar past stories when new requirements arrive (new)
4. **Augment** the LLM prompt with retrieved stories as few-shot examples (new)
5. The cycle continues -- each generation improves the knowledge base

Over time, the system learns from its own output. User-edited stories (via the feedback agent) carry higher weight, creating a quality flywheel where human corrections propagate into future generations.

---

## 2. RAG Cycle

```
                    +-----------------------+
                    |   New Requirements    |
                    |  (BRD / UI Screens)   |
                    +-----------+-----------+
                                |
                                v
                    +-----------+-----------+
                    |   RETRIEVAL           |
                    |                       |
                    |  1. Embed req text    |
                    |  2. Query pgvector    |
                    |     (cosine search)   |
                    |  3. Get top-3 similar |
                    |     past stories      |
                    +-----------+-----------+
                                |
                        similar stories
                                |
                                v
                    +-----------+-----------+
                    |   AUGMENTATION        |
                    |                       |
                    |  Inject retrieved     |
                    |  stories into Gemini  |
                    |  prompt as few-shot   |
                    |  reference examples   |
                    +-----------+-----------+
                                |
                         augmented prompt
                                |
                                v
                    +-----------+-----------+
                    |   GENERATION          |
                    |                       |
                    |  Gemini 2.5 Flash     |
                    |  generates new user   |
                    |  stories (existing    |
                    |  agent pipeline)      |
                    +-----------+-----------+
                                |
                          new stories
                                |
                                v
                    +-----------+-----------+
                    |   INSERTION           |
                    |                       |
                    |  1. Embed new stories |
                    |  2. Store in pgvector |
                    |  3. Available for     |
                    |     future retrieval  |
                    +-----------+-----------+
                                |
                                v
                    +-----------+-----------+
                    |   pgvector / Postgres |
                    |   (Growing Knowledge  |
                    |    Base)              |
                    +-----------------------+
                                ^
                                |
                     The cycle continues...
```

---

## 3. Vector Store: pgvector

### Why pgvector (not ChromaDB or a standalone vector DB)

| Factor | pgvector | ChromaDB |
|--------|----------|----------|
| Infrastructure | PostgreSQL already configured in `db.py` | Requires new file-based store or separate process |
| ORM integration | SQLAlchemy `Vector` column type, uses existing `SessionLocal` | Separate client API, no SQLAlchemy integration |
| Transactional consistency | Story metadata + embedding in same row, same transaction | Two stores to keep in sync |
| Scale fit | Handles thousands of stories with HNSW indexing | Designed for larger scale than needed here |
| Failure modes | Single database = single failure mode | Two data stores = two failure modes |

pgvector supports cosine distance (`<=>`), L2 distance (`<->`), and inner product (`<#>`). We use **cosine distance** for normalized text embeddings.

### Database Schema

**Table: `clarifai_user_story_embeddings`**

```sql
CREATE TABLE clarifai_user_story_embeddings (
    -- Primary key
    id                  VARCHAR(64) PRIMARY KEY,

    -- User story content (stored for retrieval display)
    user_story_id       VARCHAR(20)  NOT NULL,      -- e.g. "US-001"
    title               VARCHAR(500) NOT NULL,
    user_story_text     TEXT         NOT NULL,       -- "As a... I want... so that..."
    acceptance_criteria TEXT[],                       -- Array of criteria strings
    description         TEXT,

    -- Source requirement that produced this story
    requirement_id      VARCHAR(20),                 -- e.g. "REQ-001" or "N/A"
    requirement_text    TEXT,                         -- The input requirement

    -- Metadata for filtering
    tags                TEXT[],
    priority            VARCHAR(20),                 -- High / Medium / Low
    tshirt_size         VARCHAR(5),                  -- XS / S / M / L / XL
    confidence_score    FLOAT,

    -- Source tracking
    source_agent        VARCHAR(50)  NOT NULL,       -- "UserStoryAgent" / "RequirementUserStoryAgent"
    source_doc_id       VARCHAR(64),                 -- Links to ProgressTracker.doc_id

    -- The embedding vector (768 dimensions)
    embedding           vector(768)  NOT NULL,

    -- Timestamps
    created_at          TIMESTAMP DEFAULT NOW(),

    -- Quality signals
    user_rating         INTEGER,                     -- 1-5, set via feedback
    was_edited          BOOLEAN DEFAULT FALSE        -- True if user modified after generation
);

-- HNSW index for fast cosine similarity search (create after initial data load)
CREATE INDEX idx_story_embedding_cosine
    ON clarifai_user_story_embeddings
    USING hnsw (embedding vector_cosine_ops);
```

### SQLAlchemy Model

Added to `ClarifAI/models.py` alongside the existing `ProgressTracker`:

```python
from pgvector.sqlalchemy import Vector

class UserStoryEmbedding(Base):
    __tablename__ = "clarifai_user_story_embeddings"

    id                  = Column(String(64), primary_key=True, default=lambda: uuid.uuid4().hex)
    user_story_id       = Column(String(20), nullable=False)
    title               = Column(String(500), nullable=False)
    user_story_text     = Column(Text, nullable=False)
    acceptance_criteria = Column(ARRAY(Text))
    description         = Column(Text)
    requirement_id      = Column(String(20))
    requirement_text    = Column(Text)
    tags                = Column(ARRAY(Text))
    priority            = Column(String(20))
    tshirt_size         = Column(String(5))
    confidence_score    = Column(Float)
    source_agent        = Column(String(50), nullable=False)
    source_doc_id       = Column(String(64))
    embedding           = Column(Vector(768), nullable=False)
    created_at          = Column(TIMESTAMP, default=datetime.utcnow)
    user_rating         = Column(Integer)
    was_edited          = Column(Boolean, default=False)
```

### One-Time Setup

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

Then run `Base.metadata.create_all(engine)` from Python to create the table.

---

## 4. Embedding Strategy

### Model

**Gemini `gemini-embedding-001`** via the `google-genai` SDK (already used by `agents/image_extractor_agent.py`).

- **Dimensions:** 768 (truncated from the full 3072 via `output_dimensionality` config)
- **Why 768:** Sufficient for user story semantic search, significantly reduces storage/index size, quality preserved by Matryoshka training

### What Gets Embedded

Each user story is embedded as a **single composite text string** combining the most semantically meaningful fields:

```
Title: User Authentication via OTP
Story: As a user, I want to log in with OTP so that I can access my account securely.
Acceptance Criteria:
- User receives OTP after entering username
- OTP must expire after 5 minutes
- User cannot proceed without valid OTP
Tags: authentication, security, login
```

**Why these fields:**
- `title` -- concise semantic anchor
- `user_story` -- carries core intent (role, action, benefit)
- `acceptance_criteria` -- adds behavioral specificity
- `tags` -- provides categorical signal

The original `requirement_text` is stored as metadata for display during retrieval but is **not** embedded. We embed the *output* (user story) because the RAG goal is to find similar *output examples* for few-shot prompting.

### Composition Function

```python
def compose_embedding_text(story: dict) -> str:
    parts = []
    if story.get("title"):
        parts.append(f"Title: {story['title']}")

    # Handle both agent output schemas
    story_text = story.get("user_story") or story.get("story", "")
    if story_text:
        parts.append(f"Story: {story_text}")

    criteria = story.get("acceptance_criteria") or story.get("acceptanceCriteria") or []
    if criteria:
        parts.append("Acceptance Criteria:")
        for c in criteria:
            parts.append(f"- {c}")

    tags = story.get("tags") or story.get("Tags/Labels") or []
    if tags:
        parts.append(f"Tags: {', '.join(tags)}")

    return "\n".join(parts)
```

---

## 5. Retrieval Strategy

### Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `k` (candidates retrieved from DB) | 5 | Cast a wider net from the vector store |
| `max_examples` (injected into prompt) | 3 | Only top 3 by similarity enter the prompt (~1500 tokens budget) |
| `similarity_threshold` | 0.70 | Below 0.70 cosine similarity, examples are too dissimilar and would confuse the LLM |
| `RAG_MIN_STORIES` | 10 | Retrieval is skipped until the vector store has at least 10 stories (cold start protection) |

### Search Query Construction

**For `UserStoryAgent`** (BRD pipeline):
- Concatenate all requirement texts being processed into a single query string
- This gives broad semantic coverage of the batch

**For `RequirementUserStoryAgent`** (image/UI pipeline):
- Use the first 500 characters of `combined_info` (raw_info + reference doc text)
- Captures domain context without exceeding embedding input limits

### SQL Query

```sql
SELECT *,
       1 - (embedding <=> :query_embedding) AS similarity
FROM clarifai_user_story_embeddings
WHERE 1 - (embedding <=> :query_embedding) >= :threshold
  AND (:tag_filter IS NULL OR tags && :tag_filter)   -- optional tag overlap filter
ORDER BY embedding <=> :query_embedding ASC
LIMIT :k;
```

### Metadata Filtering

When incoming requirements have identifiable tags/domains, pre-filter the vector store to retrieve examples from the same functional domain:

```python
# If requirements are about authentication, only retrieve auth-related examples
tag_filter = ["authentication", "security"]  # extracted from requirement tags
```

This ensures authentication stories are used as examples for authentication requirements, not billing stories.

### Cold Start Handling

When `get_story_count() < RAG_MIN_STORIES`:
- Skip retrieval entirely
- Generate stories using the original prompt (no augmentation)
- Still insert generated stories into the vector store
- After 10+ stories accumulate, RAG activates automatically

Optional: Use the `/rag/seed` endpoint to bulk-import historical user stories from past runs (Excel exports) to bootstrap the knowledge base.

---

## 6. Prompt Augmentation

### Injection Location

Few-shot examples are injected **between** the system instructions and the input requirements:

```
[System instructions / role description / output format]

[--- FEW-SHOT EXAMPLES SECTION ---]    <-- NEW: injected here

[Input requirements to process]
```

### Example Format

```
Below are high-quality user stories previously generated for similar requirements.
Use these as reference for style, detail level, and acceptance criteria quality.
Do NOT copy these stories -- generate original content for the new requirements.

--- Reference Example 1 (relevance: 0.92) ---
Original Requirement: "The system shall allow users to reset their password via email."
Generated User Story:
{
    "user_story_id": "US-012",
    "title": "Password Reset via Email Link",
    "user_story": "As a registered user, I want to reset my password via an email link so that I can regain access to my account.",
    "description": "Covers the password reset flow including link generation, email delivery, and password update.",
    "acceptance_criteria": [
        "Given a user is on the login page, When they click Forgot Password and enter their email, Then a reset link is sent within 60 seconds.",
        "Given a reset link has been sent, When the user clicks it within 24 hours, Then they can set a new password.",
        "Given a reset link has expired, When the user clicks it, Then an error message is shown with option to request a new link."
    ],
    "tags": ["authentication", "password", "email"],
    "tshirt_size": "M",
    "priority": "High",
    "confidence_score": 0.95
}

--- Reference Example 2 (relevance: 0.85) ---
...

Now generate user stories for the following NEW requirements:
```

### Why This Works

- The LLM sees concrete examples of what "good output" looks like for similar requirements
- Acceptance criteria quality is demonstrated (Given/When/Then format, negative scenarios)
- The "Do NOT copy" instruction prevents direct regurgitation
- The relevance score tells the LLM which examples are more applicable
- Over time, as user-edited stories enter the vector store, examples improve, creating the quality flywheel

---

## 7. Insertion Strategy

### When Stories Get Stored

Stories are inserted at three points in the application:

| Insertion Point | Agent | Entry Points Served |
|-----------------|-------|---------------------|
| After `UserStoryAgent.run()` | UserStoryAgent | CLI (`main.py`), Streamlit UI, `/user_stories` API |
| After `RequirementUserStoryAgent.run()` | RequirementUserStoryAgent | `/generate-userstory`, `/extract-and-generate` API |
| After `UserStoryFeedbackAgent.run()` | UserStoryFeedbackAgent | `/update-story` API (user-edited, higher quality) |

### Insertion Flow

```
Stories generated by LLM
    |
    v
compose_embedding_text()          --> build text for each story
    |
    v
generate_embeddings_batch()       --> call Gemini embedding API
    |
    v
store_user_stories_batch()        --> INSERT into PostgreSQL + pgvector
```

### Deduplication

Before inserting, check if a story with the same `user_story_text` already exists (exact text match). If it does, skip that story. This prevents duplicates when the same requirements are re-processed.

### Non-Blocking Insertion

All insertion logic is wrapped in `try/except`. If the embedding API or database fails, the main generation flow continues uninterrupted. A warning is logged. The user gets their stories regardless.

```python
# RAG: Store generated stories (non-blocking)
if RAG_ENABLED and parsed:
    try:
        store_generated_stories(parsed, source_agent="UserStoryAgent")
    except Exception as e:
        logging.warning(f"RAG storage failed (non-blocking): {e}")
```

### Feedback-Edited Stories

Stories modified via `UserStoryFeedbackAgent` are stored with `was_edited=True`. These carry higher quality signal and are boosted in retrieval results.

---

## 8. New Files

### 8.1 `ClarifAI/tools/embedding_utils.py`

Encapsulates all embedding generation logic using the Gemini embedding API.

```python
def compose_embedding_text(story: dict) -> str:
    """
    Build the composite text string from a user story dict.
    Combines title, user_story, acceptance_criteria, and tags.
    Handles both UserStoryAgent and RequirementUserStoryAgent output schemas.
    """

def generate_embedding(text: str, api_key: str = None, dimensions: int = 768) -> List[float]:
    """
    Generate a single embedding vector using Gemini gemini-embedding-001.
    Uses google.genai.Client.models.embed_content().
    Returns a list of 768 floats.
    """

def generate_embeddings_batch(texts: List[str], api_key: str = None, dimensions: int = 768) -> List[List[float]]:
    """
    Batch embed multiple texts in a single API call.
    Falls back to individual calls if batch fails.
    Implements exponential backoff retry (3 attempts).
    """
```

**Implementation pattern:** Uses `google.genai` client (same pattern as `agents/image_extractor_agent.py`), not the older `google.generativeai` SDK.

### 8.2 `ClarifAI/tools/vector_store.py`

All pgvector CRUD operations using SQLAlchemy and the existing `SessionLocal` from `db.py`.

```python
def store_user_story(story: dict, embedding: List[float], session=None, source_agent: str = "unknown") -> str:
    """Insert a single story + embedding. Returns the database record ID."""

def store_user_stories_batch(stories: List[dict], embeddings: List[List[float]], session=None, source_agent: str = "unknown") -> List[str]:
    """Batch insert using SQLAlchemy bulk_save_objects."""

def search_similar_stories(query_embedding: List[float], k: int = 5, similarity_threshold: float = 0.70, tag_filter: List[str] = None, session=None) -> List[dict]:
    """
    Cosine similarity search via pgvector <=> operator.
    Returns stories above threshold, ordered by similarity descending.
    Each result includes story data + similarity_score.
    """

def get_story_count(session=None) -> int:
    """Total stored stories (for cold start check)."""

def delete_stories_by_source(source_id: str, session=None) -> int:
    """Remove stories from a specific generation run."""
```

### 8.3 `ClarifAI/tools/rag_utils.py`

The RAG orchestration layer -- glue between embedding, vector store, and prompt augmentation.

```python
# Configuration (loaded from environment)
RAG_ENABLED: bool           # Default: True
RAG_MIN_STORIES: int        # Default: 10
RAG_RETRIEVAL_K: int        # Default: 5
RAG_SIMILARITY_THRESHOLD: float  # Default: 0.70
RAG_MAX_EXAMPLES: int       # Default: 3

def retrieve_similar_examples(requirement_text: str, k: int = 5, threshold: float = 0.70, tag_filter: List[str] = None) -> List[dict]:
    """
    RETRIEVAL step: Embed the requirement text, search vector store.
    Returns similar past stories with similarity scores.
    Returns empty list if RAG disabled or store has < RAG_MIN_STORIES.
    """

def format_examples_for_prompt(examples: List[dict], max_examples: int = 3) -> str:
    """
    AUGMENTATION step: Format retrieved stories into a prompt-ready string.
    Includes the original requirement, generated story JSON, and similarity score.
    Caps at max_examples to control token budget.
    """

def store_generated_stories(stories: List[dict], source_agent: str = "unknown", requirement_texts: dict = None) -> int:
    """
    INSERTION step: Embed and store generated stories.
    Handles batch embedding, deduplication, and database insertion.
    Returns count of stories stored.
    """
```

**Graceful degradation:** All functions return empty/zero results if RAG is disabled or the vector store is unreachable. The system works exactly as it does today in degraded mode.

---

## 9. Files to Modify

### 9.1 `ClarifAI/agents/user_story_agent.py`

The primary agent serving CLI, Streamlit, and `/user_stories` API.

**Changes:**
1. Import `rag_utils` at the top
2. After filtering `functional_reqs` (line 19), before building the prompt: add RAG retrieval
3. Inside the prompt template (before "Requirements:" section): conditionally inject few-shot examples
4. After successful JSON parsing (line 167), before return: add RAG insertion

```python
# RETRIEVAL (before prompt construction)
rag_examples_text = ""
if RAG_ENABLED:
    all_req_texts = " ".join([r["requirement_text"] for r in functional_reqs])
    similar = retrieve_similar_examples(all_req_texts, k=5, threshold=0.70)
    if similar:
        rag_examples_text = format_examples_for_prompt(similar, max_examples=3)

# AUGMENTATION (in prompt, before "Requirements:" section)
# ... inject rag_examples_text if non-empty ...

# INSERTION (after generation, before return)
if RAG_ENABLED and parsed:
    try:
        req_texts = {r["requirement_id"]: r["requirement_text"] for r in functional_reqs}
        store_generated_stories(parsed, source_agent="UserStoryAgent", requirement_texts=req_texts)
    except Exception as e:
        logging.warning(f"RAG storage failed: {e}")
```

### 9.2 `ClarifAI/agents/requirement_user_story_agent.py`

The combined agent for image extraction + story generation. **Easiest RAG integration point** since it already injects reference context into the prompt.

**Changes:** Same pattern as `user_story_agent.py`:
1. Import `rag_utils`
2. After building `combined_info` (line 36): retrieve similar examples using first 500 chars as query
3. Insert examples into the prompt before the `Input:` section
4. After parsing: store generated stories

### 9.3 `ClarifAI/agents/user_story_feedback_agent.py`

Stores user-edited stories as higher-quality signals.

**Changes:** After the return statement (line 80), add insertion of the updated story with `was_edited=True`:

```python
if RAG_ENABLED and updated:
    try:
        updated["was_edited"] = True
        store_generated_stories([updated], source_agent="UserStoryFeedbackAgent")
    except Exception:
        pass
```

### 9.4 `ClarifAI/models.py`

Add the `UserStoryEmbedding` SQLAlchemy model (see Section 3 for full schema).

### 9.5 `ClarifAI/db.py`

Add pgvector type registration on connection and a `init_vector_extension()` utility:

```python
from sqlalchemy import event, text

@event.listens_for(engine, "connect")
def connect(dbapi_connection, connection_record):
    from pgvector.sqlalchemy import Vector  # registers the type

def init_vector_extension():
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
```

### 9.6 `ClarifAI/requirements.txt`

Add:
```
pgvector
google-genai
```

### 9.7 `ClarifAI/.env`

Add:
```
# RAG Configuration
RAG_ENABLED=true
RAG_EMBEDDING_MODEL=gemini-embedding-001
RAG_EMBEDDING_DIMENSIONS=768
RAG_RETRIEVAL_K=5
RAG_SIMILARITY_THRESHOLD=0.70
RAG_MAX_EXAMPLES=3
RAG_MIN_STORIES_FOR_RETRIEVAL=10
```

### 9.8 `ClarifAI/api/endpoints/requirements.py`

Add new RAG-specific endpoints after existing endpoint blocks:

```python
@router.post("/rag/search")
async def rag_search_stories(request: RAGSearchRequest):
    """Semantic search for similar past user stories."""

@router.get("/rag/stats")
async def rag_stats():
    """Vector store statistics (total stories, breakdown by source agent)."""
```

### 9.9 `ClarifAI/main.py`

Add RAG storage after user story generation in the CLI pipeline (after line 52):

```python
if user_stories.get("user_stories"):
    from tools.rag_utils import store_generated_stories, RAG_ENABLED
    if RAG_ENABLED:
        try:
            store_generated_stories(user_stories["user_stories"], source_agent="CLI-UserStoryAgent")
        except Exception as e:
            print(f"RAG storage skipped: {e}")
```

### 9.10 Streamlit UIs (no changes needed)

Both `clarifAI_streamlit.py` and `clarifAI_wizard.py` call `UserStoryAgent.run()` directly. Since RAG logic is added inside the agent's `run()` method, the Streamlit UIs benefit from RAG automatically with zero code changes.

---

## 10. New API Endpoints

| Method | Path | Input | Description |
|--------|------|-------|-------------|
| POST | `/rag/search` | `{ "query": str, "k": 5, "threshold": 0.70, "tags": ["auth"] }` | Semantic search for similar past user stories. Returns matching stories with similarity scores. |
| GET | `/rag/stats` | None | Returns vector store statistics: total stories, breakdown by source agent, average confidence score. |
| POST | `/rag/seed` | `{ "user_stories": [...] }` (optional) | Bulk-import historical user stories into the vector store. Useful for bootstrapping the RAG knowledge base with existing data from Excel exports. |

---

## 11. Dependencies

Add to `requirements.txt`:

| Package | Version | Purpose |
|---------|---------|---------|
| `pgvector` | >= 0.3.0 | SQLAlchemy `Vector` column type and distance operators for PostgreSQL |
| `google-genai` | >= 1.0.0 | Unified Google GenAI SDK for `embed_content()` API |

**Not needed:**
- `sentence-transformers` -- Gemini handles embeddings natively
- `langchain` -- custom RAG orchestration is simpler and avoids framework overhead
- `chromadb` -- using pgvector in existing PostgreSQL instead

**Prerequisite:** The `pgvector` PostgreSQL extension must be installed on the database server (`CREATE EXTENSION IF NOT EXISTS vector;`).

---

## 12. Implementation Order

### Phase 1: Foundation

1. Install pgvector PostgreSQL extension
2. Create `tools/embedding_utils.py` -- implement `compose_embedding_text()` and `generate_embedding()`
3. Add `UserStoryEmbedding` model to `models.py`
4. Update `db.py` with pgvector connection setup
5. Update `requirements.txt` and `.env`
6. Run `Base.metadata.create_all(engine)` to create the table

**Verify:** Generate an embedding for a sample user story text, confirm it returns 768 floats.

### Phase 2: Storage Layer

7. Create `tools/vector_store.py` -- implement `store_user_story()`, `store_user_stories_batch()`, `search_similar_stories()`, `get_story_count()`
8. Test: insert a few sample stories, retrieve by similarity, confirm cosine distance ordering

**Verify:** Insert 5 stories, search with a related query, confirm results are ordered by relevance.

### Phase 3: RAG Orchestration

9. Create `tools/rag_utils.py` -- implement `retrieve_similar_examples()`, `format_examples_for_prompt()`, `store_generated_stories()`
10. Test the full cycle manually: insert sample stories, retrieve, verify prompt formatting

**Verify:** Call `retrieve_similar_examples("password reset")` and confirm it returns relevant auth stories.

### Phase 4: Agent Integration

11. Modify `agents/user_story_agent.py` -- add retrieval before prompt + insertion after generation
12. Modify `agents/requirement_user_story_agent.py` -- same pattern
13. Modify `agents/user_story_feedback_agent.py` -- add insertion of edited stories
14. Modify `main.py` -- add CLI insertion

**Verify:** Run the full pipeline (extract -> classify -> validate -> generate stories), confirm stories are stored in pgvector, then run again and confirm retrieved examples appear in the prompt.

### Phase 5: API and Polish

15. Add `/rag/search` and `/rag/stats` endpoints to `api/endpoints/requirements.py`
16. Create HNSW index on the embedding column (after initial data exists)
17. End-to-end testing across all three entry points (CLI, API, Streamlit)
18. Optionally seed the vector store with historical stories from past Excel exports

**Verify:** Hit `/rag/stats` to confirm story count, hit `/rag/search` with a query to confirm semantic search works.

---

## 13. Risk Mitigation

### Cold Start

The first 10 stories have no RAG examples to draw from. `RAG_MIN_STORIES=10` ensures retrieval only activates after a baseline corpus exists. Before that threshold, the system works exactly as it does today. The optional `/rag/seed` endpoint allows pre-loading historical data from Excel exports.

### Embedding API Rate Limits

The Gemini embedding API has known rate-limit issues on batch requests. `embedding_utils.py` implements exponential backoff retry (3 attempts: 1s, 2s, 4s delays). If all retries fail, RAG insertion is skipped silently and a warning is logged. The main generation flow is never blocked.

### Prompt Length

Adding 3 few-shot examples adds approximately 1500 tokens. Gemini 2.5 Flash has a 1M token context window, so this is negligible. The `RAG_MAX_EXAMPLES` parameter in `.env` allows reducing this if needed.

### Quality Drift

If early low-quality stories contaminate the RAG examples, they could degrade future output. Mitigations:
- **Confidence filter:** Only retrieve stories with `confidence_score >= 0.80`
- **Edited boost:** Stories modified via feedback (`was_edited=True`) are ranked higher
- **Threshold guard:** The 0.70 similarity threshold prevents weakly-related stories from entering the prompt
- **User rating (future):** A 1-5 rating field allows manual quality curation

### Database Migration

Adding a new table is purely additive. `Base.metadata.create_all(engine)` only creates tables that don't exist yet -- it does not modify the existing `clarifai_progress_tracker` table.

### Graceful Degradation

Every RAG operation (retrieval, embedding, storage) is wrapped in try/except with fallback to the existing non-RAG behavior. If PostgreSQL is down, the embedding API is unreachable, or `RAG_ENABLED=false`, the application works identically to today. RAG is an enhancement layer, not a dependency.

---

## 14. Configuration Reference

All RAG behavior is controlled via environment variables in `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `RAG_ENABLED` | `true` | Master switch. Set to `false` to disable all RAG behavior. |
| `RAG_EMBEDDING_MODEL` | `gemini-embedding-001` | Gemini embedding model to use |
| `RAG_EMBEDDING_DIMENSIONS` | `768` | Output dimensionality for embeddings |
| `RAG_RETRIEVAL_K` | `5` | Number of candidates to retrieve from vector store |
| `RAG_SIMILARITY_THRESHOLD` | `0.70` | Minimum cosine similarity for retrieved examples |
| `RAG_MAX_EXAMPLES` | `3` | Maximum examples injected into the LLM prompt |
| `RAG_MIN_STORIES_FOR_RETRIEVAL` | `10` | Minimum stories in DB before retrieval activates |

---

## 15. File Summary

### New Files

| File | Purpose |
|------|---------|
| `ClarifAI/tools/embedding_utils.py` | Gemini embedding API wrapper (compose text, generate embeddings) |
| `ClarifAI/tools/vector_store.py` | pgvector CRUD operations (store, search, count, delete) |
| `ClarifAI/tools/rag_utils.py` | RAG orchestration (retrieve, format, augment, store) |

### Modified Files

| File | Change |
|------|--------|
| `ClarifAI/models.py` | Add `UserStoryEmbedding` SQLAlchemy model |
| `ClarifAI/db.py` | Add pgvector type registration, `init_vector_extension()` |
| `ClarifAI/agents/user_story_agent.py` | Add RAG retrieval before prompt, insertion after generation |
| `ClarifAI/agents/requirement_user_story_agent.py` | Same RAG integration pattern |
| `ClarifAI/agents/user_story_feedback_agent.py` | Store edited stories with `was_edited=True` |
| `ClarifAI/main.py` | Add RAG storage in CLI pipeline |
| `ClarifAI/api/endpoints/requirements.py` | Add `/rag/search`, `/rag/stats` endpoints |
| `ClarifAI/requirements.txt` | Add `pgvector`, `google-genai` |
| `ClarifAI/.env` | Add RAG configuration variables |

### Unchanged Files

| File | Why Unchanged |
|------|---------------|
| `ClarifAI/clarifAI_streamlit.py` | Calls `UserStoryAgent.run()` -- RAG is inside the agent |
| `ClarifAI/clarifAI_wizard.py` | Same -- benefits from RAG automatically |
| `ClarifAI/tools/llm_utils.py` | LLM call interface unchanged; prompts are augmented before calling |
| `ClarifAI/tools/prompt_utils.py` | JSON parsing unchanged |
| All other agents | Only user-story-generating agents need RAG |
