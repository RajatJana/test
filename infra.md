# ClarifAI RAG — Infrastructure Setup & Activation Guide

> Complete step-by-step instructions for setting up PostgreSQL, pgvector, Python dependencies,
> and activating the RAG (Retrieval-Augmented Generation) pipeline on Windows 11.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Install PostgreSQL](#2-install-postgresql)
3. [Install the pgvector Extension](#3-install-the-pgvector-extension)
4. [Create the Database](#4-create-the-database)
5. [Enable the pgvector Extension in the Database](#5-enable-the-pgvector-extension-in-the-database)
6. [Install Python Dependencies](#6-install-python-dependencies)
7. [Configure Environment Variables](#7-configure-environment-variables)
8. [Update the Database Connection String](#8-update-the-database-connection-string)
9. [Create Database Tables](#9-create-database-tables)
10. [Verify Embedding Generation](#10-verify-embedding-generation)
11. [Verify Vector Store Operations](#11-verify-vector-store-operations)
12. [Seed the Vector Store](#12-seed-the-vector-store)
13. [Create HNSW Index](#13-create-hnsw-index)
14. [End-to-End RAG Flow Verification](#14-end-to-end-rag-flow-verification)
15. [Troubleshooting](#15-troubleshooting)
16. [Quick Reference — Complete Checklist](#16-quick-reference--complete-checklist)

---

## 1. Prerequisites

Before starting, confirm the following are available on your machine:

| Requirement | Why It's Needed |
|-------------|----------------|
| **Windows 11** | This guide is written for Windows 11. Commands use PowerShell. |
| **Python 3.10+** | ClarifAI requires Python 3.10 or higher. |
| **pip** | Python package installer (ships with Python). |
| **A valid Google Gemini API key** | Must have access to both `gemini-2.5-flash` (LLM) and `gemini-embedding-001` (embeddings). |
| **Internet access** | Required for downloading PostgreSQL, Python packages, and calling Gemini APIs. |
| **Admin / elevated privileges** | Required for installing PostgreSQL and its extensions. |

### Check Python version

```powershell
python --version
```

Expected output: `Python 3.10.x` or higher.

### Check pip

```powershell
pip --version
```

---

## 2. Install PostgreSQL

PostgreSQL is the relational database that stores RAG vector embeddings via the pgvector extension. ClarifAI requires **PostgreSQL 14 or higher** (16+ recommended).

### Option A — Windows Installer (Recommended for first-time users)

1. Go to https://www.postgresql.org/download/windows/
2. Click **"Download the installer"** (provided by EDB).
3. Download the **PostgreSQL 16** installer (or the latest stable version).
4. Run the installer:
   - **Installation Directory**: Accept the default (`C:\Program Files\PostgreSQL\16`).
   - **Components**: Select all — especially **PostgreSQL Server**, **pgAdmin 4**, and **Command Line Tools**.
   - **Data Directory**: Accept the default.
   - **Password**: Set a strong password for the `postgres` superuser. **Write this down** — you will need it in Step 8.
   - **Port**: Accept the default `5432`.
   - **Locale**: Accept the default.
5. Click **Next** through the remaining screens and finish the installation.
6. When prompted to launch **Stack Builder**, you may close it (pgvector is installed separately in Step 3).

### Option B — Chocolatey

If you have Chocolatey installed:

```powershell
choco install postgresql16 --params '/Password:YourPasswordHere'
```

Replace `YourPasswordHere` with your desired `postgres` user password.

### Option C — Docker (Recommended for isolation and simplicity)

This option runs PostgreSQL inside a Docker container. The `pgvector/pgvector` image comes with pgvector **pre-installed**, so you can **skip Step 3** entirely.

1. Ensure Docker Desktop is installed and running.
2. Pull and start the container:

```powershell
docker run -d `
  --name clarifai-pg `
  -p 5432:5432 `
  -e POSTGRES_PASSWORD=YourPasswordHere `
  -e POSTGRES_DB=clarifai_system `
  -v clarifai_pgdata:/var/lib/postgresql/data `
  pgvector/pgvector:pg16
```

| Flag | Purpose |
|------|---------|
| `--name clarifai-pg` | Container name for easy reference |
| `-p 5432:5432` | Map container port 5432 to host port 5432 |
| `-e POSTGRES_PASSWORD=...` | Set the `postgres` user password |
| `-e POSTGRES_DB=clarifai_system` | Automatically create the `clarifai_system` database |
| `-v clarifai_pgdata:/var/lib/...` | Persist data across container restarts |
| `pgvector/pgvector:pg16` | PostgreSQL 16 image with pgvector pre-installed |

### Verification — PostgreSQL is running

```powershell
# If installed natively:
& "C:\Program Files\PostgreSQL\16\bin\psql.exe" -U postgres -c "SELECT version();"

# If using Docker:
docker exec -it clarifai-pg psql -U postgres -c "SELECT version();"
```

Expected: A row showing `PostgreSQL 16.x ...` (or your installed version).

> **Tip**: Add `C:\Program Files\PostgreSQL\16\bin` to your system `PATH` so you can run `psql` directly from any terminal. To do this:
> 1. Open **Start** > search "Environment Variables" > click **"Edit the system environment variables"**.
> 2. Click **Environment Variables** > under **System variables**, select **Path** > click **Edit**.
> 3. Click **New** and add: `C:\Program Files\PostgreSQL\16\bin`
> 4. Click **OK** on all dialogs. Restart your terminal.

---

## 3. Install the pgvector Extension

> **Skip this entire step if you used Docker Option C** in Step 2 — pgvector is already included in the `pgvector/pgvector` image.

pgvector is a PostgreSQL extension that adds support for vector data types and similarity search operators. It must be installed at the PostgreSQL server level (not just via pip).

### Option A — Build from source (Windows)

**Prerequisites**: Visual Studio Build Tools with the **"Desktop development with C++"** workload.

1. Install Visual Studio Build Tools (if not already installed):
   - Download from https://visualstudio.microsoft.com/visual-cpp-build-tools/
   - In the installer, select **"Desktop development with C++"**.
   - Complete the installation.

2. Open a **"Developer Command Prompt for VS"** (search for it in Start).

3. Clone and build pgvector:

```cmd
git clone --branch v0.8.0 https://github.com/pgvector/pgvector.git
cd pgvector

set "PG_CONFIG=C:\Program Files\PostgreSQL\16\bin\pg_config.exe"
nmake /F Makefile.win
nmake /F Makefile.win install
```

> Adjust the `PG_CONFIG` path if your PostgreSQL is installed in a different location or version.

### Option B — Pre-built binaries

1. Go to https://github.com/pgvector/pgvector/releases
2. Download the Windows binary matching your PostgreSQL version (e.g., `pgvector-v0.8.0-pg16-windows-x64.zip`).
3. Extract the archive and copy the files:

```powershell
# Adjust paths to match your PostgreSQL installation directory
Copy-Item vector.dll "C:\Program Files\PostgreSQL\16\lib\"
Copy-Item vector.control "C:\Program Files\PostgreSQL\16\share\extension\"
Copy-Item vector--*.sql "C:\Program Files\PostgreSQL\16\share\extension\"
```

### Verification — pgvector is installed on the server

```powershell
psql -U postgres -c "CREATE EXTENSION IF NOT EXISTS vector;" -d postgres
```

If this runs without error, pgvector is correctly installed. (You can drop it with `DROP EXTENSION vector;` afterward — we will enable it in the actual database in Step 5.)

---

## 4. Create the Database

Create the `clarifai_system` database that ClarifAI uses for both progress tracking and RAG vector storage.

> **Skip this step if you used Docker Option C** with `-e POSTGRES_DB=clarifai_system` — the database was created automatically.

```powershell
psql -U postgres -c "CREATE DATABASE clarifai_system;"
```

You will be prompted for the `postgres` user password you set during installation.

### Verification

```powershell
psql -U postgres -c "\l" | findstr clarifai_system
```

Expected: A row showing `clarifai_system` in the database list.

---

## 5. Enable the pgvector Extension in the Database

The `vector` extension must be enabled **inside** the `clarifai_system` database (creating it in `postgres` or any other database does not carry over).

```powershell
psql -U postgres -d clarifai_system -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

### For Docker

```powershell
docker exec -it clarifai-pg psql -U postgres -d clarifai_system -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

### Verification

```powershell
psql -U postgres -d clarifai_system -c "SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';"
```

Expected output:

```
 extname | extversion
---------+------------
 vector  | 0.8.0
```

The version number may differ — any version `>= 0.5.0` is compatible.

---

## 6. Install Python Dependencies

ClarifAI's RAG flow requires three Python packages beyond the base dependencies:

| Package | Purpose |
|---------|---------|
| `pgvector` | SQLAlchemy integration for pgvector data types (`Vector(768)`) |
| `google-genai` | Google Generative AI SDK for the Gemini embedding API |
| `psycopg2-binary` | PostgreSQL adapter for Python / SQLAlchemy |

### Install all dependencies

```powershell
cd c:\Clarify\ClarifAI\ClarifAI
pip install -r requirements.txt
```

This installs everything listed in `requirements.txt`, which already includes `pgvector` and `google-genai`.

### Install psycopg2-binary explicitly

`psycopg2-binary` is not listed in `requirements.txt` but is required by SQLAlchemy to connect to PostgreSQL:

```powershell
pip install psycopg2-binary
```

### Verification

Run each of these — all three should print "OK":

```powershell
python -c "from pgvector.sqlalchemy import Vector; print('pgvector OK')"
python -c "from google import genai; print('google-genai OK')"
python -c "import psycopg2; print('psycopg2 OK')"
```

### Troubleshooting dependency issues

| Error | Fix |
|-------|-----|
| `pip install psycopg2-binary` fails | Try `pip install psycopg2` instead, or ensure Visual C++ Build Tools are installed |
| `ModuleNotFoundError: No module named 'pgvector'` | Run `pip install pgvector` again, ensure you're using the correct Python environment |
| `google-genai` conflicts with `google-generativeai` | Both can coexist — they are separate packages. `google-genai` provides `google.genai.Client`, `google-generativeai` provides `google.generativeai` |

---

## 7. Configure Environment Variables

**File:** `ClarifAI/ClarifAI/.env`

This file contains API keys and RAG configuration. The RAG settings are already present with correct defaults. You need to **replace the placeholder API keys** with real values.

### Current contents of `.env`

```env
GOOGLE_API_KEY_1= YOUR_GEMINI_API_KEY       # <-- Replace
GOOGLE_API_KEY_2= YOUR_GEMINI_API_KEY       # <-- Replace
GOOGLE_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/

GEMINI_API_KEY = YOUR_GEMINI_API_KEY        # <-- Replace

# RAG Configuration
RAG_ENABLED=true                            # Master switch for RAG
RAG_EMBEDDING_MODEL=gemini-embedding-001    # Embedding model
RAG_EMBEDDING_DIMENSIONS=768               # Vector dimensions
RAG_RETRIEVAL_K=5                          # Candidates to retrieve
RAG_SIMILARITY_THRESHOLD=0.70             # Minimum cosine similarity
RAG_MAX_EXAMPLES=3                        # Max examples in prompt
RAG_MIN_STORIES_FOR_RETRIEVAL=10          # Cold start threshold
```

### What to change

Replace **all three** occurrences of `YOUR_GEMINI_API_KEY` with your actual Gemini API key:

```env
GOOGLE_API_KEY_1=AIzaSy...your-real-key-here
GOOGLE_API_KEY_2=AIzaSy...your-real-key-here
GEMINI_API_KEY=AIzaSy...your-real-key-here
```

> **Note:** `GOOGLE_API_KEY_1` and `GOOGLE_API_KEY_2` are used for LLM calls (key rotation). `GEMINI_API_KEY` is used specifically for embedding generation. They can all be the same key, or different keys if you want separate quota pools.

### How to get a Gemini API key

1. Go to https://aistudio.google.com/apikey
2. Click **"Create API Key"**.
3. Select or create a Google Cloud project.
4. Copy the generated key (starts with `AIzaSy...`).
5. Verify the key supports the required models:
   - `gemini-2.5-flash` — used by all agents for LLM generation
   - `gemini-embedding-001` — used for RAG embedding generation

### RAG configuration reference

| Variable | Default | Description |
|----------|---------|-------------|
| `RAG_ENABLED` | `true` | Set to `false` to completely disable RAG. All agents will run without retrieval or storage. |
| `RAG_EMBEDDING_MODEL` | `gemini-embedding-001` | The Gemini model used for generating 768-dimensional embedding vectors. |
| `RAG_EMBEDDING_DIMENSIONS` | `768` | Dimensionality of embedding vectors. Must match the `Vector(N)` column in the database. |
| `RAG_RETRIEVAL_K` | `5` | Number of candidate stories to retrieve from the vector store during similarity search. |
| `RAG_SIMILARITY_THRESHOLD` | `0.70` | Minimum cosine similarity score (0.0–1.0) for a retrieved story to be included. |
| `RAG_MAX_EXAMPLES` | `3` | Maximum number of retrieved stories injected into the LLM prompt as few-shot examples. |
| `RAG_MIN_STORIES_FOR_RETRIEVAL` | `10` | Minimum stories that must exist in the vector store before retrieval activates (cold start protection). |

---

## 8. Update the Database Connection String

**File:** `ClarifAI/ClarifAI/db.py`

The connection string on **line 5** must match your PostgreSQL credentials:

```python
DATABASE_URL = "postgresql://postgres:YOURUSERNAME@localhost:5432/clarifai_system"
```

### Update it

Replace `YOURUSERNAME` with the actual password you set for the `postgres` user during PostgreSQL installation:

```python
DATABASE_URL = "postgresql://postgres:MyActualPassword123@localhost:5432/clarifai_system"
```

### Connection string format

```
postgresql://<user>:<password>@<host>:<port>/<database>
```

| Part | Default | Description |
|------|---------|-------------|
| `user` | `postgres` | PostgreSQL username |
| `password` | *(set during install)* | PostgreSQL password |
| `host` | `localhost` | Database server host |
| `port` | `5432` | PostgreSQL port |
| `database` | `clarifai_system` | Database name (created in Step 4) |

### Special characters in password

If your password contains special characters (`@`, `#`, `%`, `/`, etc.), URL-encode them:

| Character | Encoded |
|-----------|---------|
| `@` | `%40` |
| `#` | `%23` |
| `%` | `%25` |
| `/` | `%2F` |
| `:` | `%3A` |

Example: If your password is `p@ss#word`, the connection string becomes:

```python
DATABASE_URL = "postgresql://postgres:p%40ss%23word@localhost:5432/clarifai_system"
```

### Verification — Test the connection

```powershell
cd c:\Clarify\ClarifAI\ClarifAI
python -c "from db import engine; conn = engine.connect(); print('Database connection successful'); conn.close()"
```

Expected: `Database connection successful`

---

## 9. Create Database Tables

This step creates the two tables ClarifAI uses:

| Table | Purpose |
|-------|---------|
| `clarifai_progress_tracker` | Tracks document processing progress (doc_id, stage, output paths) |
| `clarifai_user_story_embeddings` | RAG vector store (user story content, 768-dim embedding, metadata, quality signals) |

### Run the table creation

```powershell
cd c:\Clarify\ClarifAI\ClarifAI
python -c "from models import Base; from db import engine; Base.metadata.create_all(engine); print('Tables created successfully')"
```

Expected: `Tables created successfully`

### Verification — Inspect the RAG table schema

```powershell
psql -U postgres -d clarifai_system -c "\d clarifai_user_story_embeddings"
```

Expected output (column list):

```
          Column          |            Type             |         Default
--------------------------+-----------------------------+-------------------------
 id                       | character varying(64)       |
 user_story_id            | character varying(20)       |
 title                    | character varying(500)      |
 user_story_text          | text                        |
 acceptance_criteria      | text[]                      |
 description              | text                        |
 requirement_id           | character varying(20)       |
 requirement_text         | text                        |
 tags                     | text[]                      |
 priority                 | character varying(20)       |
 tshirt_size              | character varying(5)        |
 confidence_score         | double precision            |
 source_agent             | character varying(50)       |
 source_doc_id            | character varying(64)       |
 embedding                | vector(768)                 |
 created_at               | timestamp without time zone |
 user_rating              | integer                     |
 was_edited               | boolean                     |
```

The key column is `embedding` with type `vector(768)` — this confirms pgvector is correctly integrated.

### Also verify the progress tracker table

```powershell
psql -U postgres -d clarifai_system -c "\d clarifai_progress_tracker"
```

---

## 10. Verify Embedding Generation

This step confirms that the Gemini embedding API is accessible and produces vectors of the correct dimensionality.

```powershell
cd c:\Clarify\ClarifAI\ClarifAI
python -c "
from tools.embedding_utils import generate_embedding, compose_embedding_text

sample_story = {
    'title': 'Login via OTP',
    'user_story': 'As a user I want to login via OTP so that I can access my account securely',
    'acceptance_criteria': ['Given a valid phone number, When OTP is requested, Then a 6-digit code is sent'],
    'tags': ['authentication', 'security']
}

text = compose_embedding_text(sample_story)
print(f'Composed text:\n{text}\n')

embedding = generate_embedding(text)
print(f'Embedding dimensions: {len(embedding)}')
print(f'First 5 values: {embedding[:5]}')
print(f'Last 5 values: {embedding[-5:]}')
print('Embedding generation OK')
"
```

### Expected output

```
Composed text:
Title: Login via OTP
Story: As a user I want to login via OTP so that I can access my account securely
Acceptance Criteria:
- Given a valid phone number, When OTP is requested, Then a 6-digit code is sent
Tags: authentication, security

Embedding dimensions: 768
First 5 values: [0.0123, -0.0456, ...]
Last 5 values: [...]
Embedding generation OK
```

### What to check

- `Embedding dimensions: 768` — must be exactly 768.
- Values are floating-point numbers (positive and negative).
- No errors or exceptions.

### If this fails

| Error | Cause | Fix |
|-------|-------|-----|
| `google.api_core.exceptions.InvalidArgument` | Invalid API key or unsupported model | Verify `GEMINI_API_KEY` in `.env` is correct and has access to `gemini-embedding-001` |
| `google.api_core.exceptions.ResourceExhausted` (429) | API rate limit hit | Wait a minute and retry — the built-in retry logic (3 attempts with backoff) should handle transient rate limits |
| `ModuleNotFoundError: google.genai` | `google-genai` not installed | Run `pip install google-genai` |

---

## 11. Verify Vector Store Operations

This step tests the full cycle: insert a story into the vector store, then search for it by semantic similarity.

```powershell
cd c:\Clarify\ClarifAI\ClarifAI
python -c "
from tools.vector_store import store_user_story, search_similar_stories, get_story_count
from tools.embedding_utils import generate_embedding, compose_embedding_text

# --- INSERT ---
story = {
    'user_story_id': 'US-TEST-001',
    'title': 'Password Reset via Email',
    'user_story': 'As a registered user, I want to reset my password via email so that I can regain access to my account',
    'acceptance_criteria': [
        'Given I click Forgot Password, When I enter my email, Then a reset link is sent within 30 seconds',
        'Given I click the reset link, When I enter a new password, Then my password is updated'
    ],
    'tags': ['authentication', 'password', 'email'],
    'priority': 'High',
    'tshirt_size': 'M',
    'confidence_score': 0.92,
    'requirement_id': 'REQ-TEST-001'
}

text = compose_embedding_text(story)
embedding = generate_embedding(text)
record_id = store_user_story(story, embedding, source_agent='infra-test')
print(f'INSERT OK - Record ID: {record_id}')

# --- SEARCH ---
query_embedding = generate_embedding('password recovery and account access')
results = search_similar_stories(query_embedding, k=3, similarity_threshold=0.5)
print(f'SEARCH OK - Found {len(results)} result(s)')
for r in results:
    print(f'  Title: {r[\"title\"]}')
    print(f'  Similarity: {r[\"similarity_score\"]:.4f}')
    print(f'  Story: {r[\"user_story_text\"][:80]}...')
    print()

# --- COUNT ---
count = get_story_count()
print(f'COUNT OK - Total stories in vector store: {count}')
"
```

### Expected output

```
INSERT OK - Record ID: <32-character hex string>
SEARCH OK - Found 1 result(s)
  Title: Password Reset via Email
  Similarity: 0.8xxx
  Story: As a registered user, I want to reset my password via email so that I can regai...

COUNT OK - Total stories in vector store: 1
```

### What to check

- **INSERT**: Returns a hex record ID (not `None`).
- **SEARCH**: Returns at least 1 result with a similarity score above 0.5.
- **COUNT**: Shows 1 (or more, if you've run this before).
- No database connection errors.

### Clean up test data (optional)

```powershell
python -c "
from tools.vector_store import delete_stories_by_source
deleted = delete_stories_by_source('infra-test')
print(f'Cleaned up {deleted} test stories')
"
```

---

## 12. Seed the Vector Store

The RAG retrieval system has a **cold start protection**: it will not retrieve examples until the vector store contains at least **10 stories** (configurable via `RAG_MIN_STORIES_FOR_RETRIEVAL`). This prevents low-quality few-shot examples when the knowledge base is too small.

You must seed the store with initial data before RAG retrieval activates.

### Option A — Seed via API (bulk import)

1. Start the FastAPI server:

```powershell
cd c:\Clarify\ClarifAI\ClarifAI
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

2. POST a batch of historical user stories to the `/rag/seed` endpoint:

```powershell
curl -X POST http://localhost:8000/rag/seed `
  -H "Content-Type: application/json" `
  -d '{
    \"user_stories\": [
      {
        \"user_story_id\": \"US-SEED-001\",
        \"title\": \"User Login with Email\",
        \"user_story\": \"As a registered user, I want to log in with my email and password so that I can access my account\",
        \"acceptance_criteria\": [\"Given valid credentials, When I submit the login form, Then I am redirected to the dashboard\"],
        \"tags\": [\"authentication\"],
        \"priority\": \"High\",
        \"tshirt_size\": \"S\",
        \"confidence_score\": 0.95
      },
      {
        \"user_story_id\": \"US-SEED-002\",
        \"title\": \"User Registration\",
        \"user_story\": \"As a new user, I want to register an account so that I can use the platform\",
        \"acceptance_criteria\": [\"Given I fill in all required fields, When I submit the form, Then my account is created\"],
        \"tags\": [\"registration\"],
        \"priority\": \"High\",
        \"tshirt_size\": \"M\",
        \"confidence_score\": 0.90
      }
    ]
  }'
```

> Repeat with more stories until you reach 10+. Use real user stories from past BRD processing runs for best quality.

### Option B — Run the pipeline with sample documents

Each pipeline run generates user stories that are automatically stored in the vector store. Run the pipeline with different BRD documents until 10+ stories accumulate.

```powershell
cd c:\Clarify\ClarifAI\ClarifAI
python main.py
```

Sample documents are available in the `sample_docs/` directory.

### Option C — Lower the cold start threshold (for testing only)

Temporarily set the minimum stories to 1 in `.env`:

```env
RAG_MIN_STORIES_FOR_RETRIEVAL=1
```

This allows RAG retrieval to activate with even a single story in the store. **Raise it back to 10 for production** to ensure sufficient example quality.

### Verification — Check story count

```powershell
# Via API (if server is running)
curl http://localhost:8000/rag/stats

# Via Python
cd c:\Clarify\ClarifAI\ClarifAI
python -c "from tools.vector_store import get_story_count; print(f'Stories in vector store: {get_story_count()}')"
```

Expected: `total_stories_in_vector_store` >= 10 (or your configured threshold).

---

## 13. Create HNSW Index

HNSW (Hierarchical Navigable Small World) is an indexing algorithm that dramatically speeds up approximate nearest-neighbor search on vector columns. Without this index, every similarity search performs a full sequential scan of all embeddings.

> **Important:** Create this index **after** the vector store has data (after Step 12). HNSW indexes are built from existing data — creating one on an empty table produces a degenerate index that won't benefit future inserts.

### Create the index

```powershell
psql -U postgres -d clarifai_system -c "
CREATE INDEX IF NOT EXISTS idx_story_embedding_cosine
  ON clarifai_user_story_embeddings
  USING hnsw (embedding vector_cosine_ops);
"
```

For Docker:

```powershell
docker exec -it clarifai-pg psql -U postgres -d clarifai_system -c "
CREATE INDEX IF NOT EXISTS idx_story_embedding_cosine
  ON clarifai_user_story_embeddings
  USING hnsw (embedding vector_cosine_ops);
"
```

### What this does

| Detail | Value |
|--------|-------|
| Index name | `idx_story_embedding_cosine` |
| Index type | HNSW (approximate nearest neighbor) |
| Operator class | `vector_cosine_ops` (optimized for cosine distance `<=>`) |
| Target column | `embedding` (`vector(768)`) |

### Verification

```powershell
psql -U postgres -d clarifai_system -c "\di idx_story_embedding_cosine"
```

Expected: One row showing the index name, table, and type.

### Performance impact

| Scenario | Without Index | With HNSW Index |
|----------|--------------|-----------------|
| 100 stories | ~1ms (scan is fast) | ~1ms |
| 1,000 stories | ~10ms | ~1ms |
| 10,000 stories | ~100ms | ~2ms |
| 100,000 stories | ~1s+ | ~5ms |

The index is most impactful as the vector store grows. For small datasets (<1,000 stories), the difference is negligible.

---

## 14. End-to-End RAG Flow Verification

This is the final validation that the complete RAG cycle works: **Retrieve -> Augment -> Generate -> Insert**.

### 14a. First pipeline run — generates and stores stories

```powershell
cd c:\Clarify\ClarifAI\ClarifAI
python main.py
```

Check the log file at `logs/ClarifAI.log` for:

```
[rag_utils] Stored X/Y stories from CLI-UserStoryAgent
```

This confirms stories were embedded and inserted into the vector store.

### 14b. Second pipeline run — retrieves past stories as examples

```powershell
python main.py
```

Check logs for:

```
[rag_utils] Retrieved N similar examples for query: ...
```

This confirms the retrieval step is working. The retrieved stories were injected into the LLM prompt as few-shot examples, improving output quality.

> If you see `[rag_utils] Cold start: only X/10 stories in store. Skipping retrieval.` instead, the store doesn't have enough stories yet. Run more pipeline iterations or seed via `/rag/seed`.

### 14c. API endpoint verification

Start the API server:

```powershell
cd c:\Clarify\ClarifAI\ClarifAI
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Test each RAG endpoint:

**1. Check RAG stats:**

```powershell
curl http://localhost:8000/rag/stats
```

Expected:

```json
{
  "total_stories_in_vector_store": 15,
  "rag_enabled": true
}
```

**2. Semantic search:**

```powershell
curl -X POST http://localhost:8000/rag/search `
  -H "Content-Type: application/json" `
  -d '{"query": "user authentication and login", "k": 3, "threshold": 0.60}'
```

Expected: A JSON response with `similar_stories` array containing matched stories with `similarity_score` values.

**3. Seed endpoint (if needed):**

```powershell
curl -X POST http://localhost:8000/rag/seed `
  -H "Content-Type: application/json" `
  -d '{"user_stories": [{"user_story_id": "US-001", "title": "Test", "user_story": "As a user I want to test", "tags": ["test"], "priority": "Low", "tshirt_size": "XS", "confidence_score": 0.8}]}'
```

Expected: `{"stored_count": 1, "total_submitted": 1}`

### 14d. Streamlit UI verification

```powershell
cd c:\Clarify\ClarifAI\ClarifAI
streamlit run clarifAI_streamlit.py
```

1. Open the URL shown in the terminal (typically `http://localhost:8501`).
2. Upload a BRD document (`.docx` or `.pdf`) in the **Extractor** tab.
3. Progress through all tabs: Extractor -> Classifier -> Validator -> **User Stories** -> Gherkin -> Test Cases -> Traceability.
4. The RAG flow is transparent — it runs inside the `UserStoryAgent` without any UI changes. Check `logs/ClarifAI.log` to confirm retrieval and storage occurred during the **User Stories** tab processing.

### Verification signals — what success looks like

| What to Check | Expected Result |
|---------------|-----------------|
| `/rag/stats` returns count > 0 | Stories are being stored in the vector store |
| Logs show `Retrieved N similar examples` on 2nd+ run | RAG retrieval is active and finding matches |
| `/rag/search` returns results with similarity scores | Cosine similarity search is working via pgvector |
| Pipeline runs successfully with `RAG_ENABLED=false` | The system gracefully degrades (identical to pre-RAG behavior) |
| Streamlit UI works without errors | RAG is transparent to the UI layer |

---

## 15. Troubleshooting

### Database connection issues

| Error | Cause | Fix |
|-------|-------|-----|
| `psycopg2.OperationalError: could not connect to server` | PostgreSQL is not running | Start the PostgreSQL service: `net start postgresql-x64-16` (or `docker start clarifai-pg` for Docker) |
| `psycopg2.OperationalError: FATAL: password authentication failed` | Wrong password in `db.py` | Update `DATABASE_URL` in `db.py` with the correct `postgres` user password |
| `psycopg2.OperationalError: FATAL: database "clarifai_system" does not exist` | Database not created | Run Step 4: `psql -U postgres -c "CREATE DATABASE clarifai_system;"` |
| `sqlalchemy.exc.OperationalError: ... connection refused` | PostgreSQL on wrong port or not listening | Verify port 5432 is open: `netstat -an | findstr 5432` |

### pgvector issues

| Error | Cause | Fix |
|-------|-------|-----|
| `ERROR: could not open extension control file ... vector.control` | pgvector not installed on PostgreSQL server | Follow Step 3 to install pgvector at the server level |
| `ERROR: extension "vector" is not available` | Same as above | Follow Step 3 |
| `column "embedding" is of type vector but expression is of type text` | pgvector type not registered in SQLAlchemy session | Verify `db.py` has the `_register_pgvector` event listener on line 11 |
| `sqlalchemy.exc.ProgrammingError: ... type "vector" does not exist` | `vector` extension not enabled in the database | Run Step 5: `CREATE EXTENSION IF NOT EXISTS vector;` in `clarifai_system` |

### Python dependency issues

| Error | Cause | Fix |
|-------|-------|-----|
| `ModuleNotFoundError: No module named 'pgvector'` | Python package not installed | `pip install pgvector` |
| `ModuleNotFoundError: No module named 'google.genai'` | `google-genai` not installed | `pip install google-genai` |
| `ModuleNotFoundError: No module named 'psycopg2'` | PostgreSQL adapter not installed | `pip install psycopg2-binary` |
| `ImportError: DLL load failed` for psycopg2 | Missing Visual C++ runtime | Install [Visual C++ Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist) |

### Embedding and RAG issues

| Error | Cause | Fix |
|-------|-------|-----|
| `generate_embedding` fails with `InvalidArgument` | Bad API key or wrong model name | Verify `GEMINI_API_KEY` in `.env` and that it has access to `gemini-embedding-001` |
| `generate_embedding` fails with 429 (rate limit) | Gemini API quota exceeded | Built-in retry (3 attempts with 1s/2s/4s backoff) handles transient limits. For persistent issues, wait or use a different API key |
| RAG retrieval returns empty list | Fewer than 10 stories in vector store | Either seed more stories (Step 12) or lower `RAG_MIN_STORIES_FOR_RETRIEVAL` in `.env` |
| RAG examples not appearing in LLM prompt | `rag_examples_text` is empty string | Check logs for cold start message or retrieval errors. Verify similarity threshold isn't too high |
| `RAG_ENABLED` is `true` but RAG does nothing | `.env` not loaded by the application | Ensure `dotenv` loads `.env` at startup, or set variables as system environment variables |
| Stories stored but never retrieved | Similarity scores below threshold | Lower `RAG_SIMILARITY_THRESHOLD` from `0.70` to `0.50` in `.env` |

### PostgreSQL service management (Windows)

```powershell
# Check if PostgreSQL service is running
Get-Service -Name "postgresql*"

# Start the service
net start postgresql-x64-16

# Stop the service
net stop postgresql-x64-16

# For Docker
docker start clarifai-pg
docker stop clarifai-pg
docker logs clarifai-pg
```

---

## 16. Quick Reference — Complete Checklist

Use this checklist to track your progress through the setup:

| # | Step | Command / Action | Status |
|---|------|-----------------|--------|
| 1 | Install PostgreSQL 16+ | Installer / Chocolatey / Docker | [ ] |
| 2 | Install pgvector extension | Build from source / pre-built binaries (skip for Docker) | [ ] |
| 3 | Create `clarifai_system` database | `psql -U postgres -c "CREATE DATABASE clarifai_system;"` | [ ] |
| 4 | Enable `vector` extension in database | `psql -U postgres -d clarifai_system -c "CREATE EXTENSION IF NOT EXISTS vector;"` | [ ] |
| 5 | Install Python packages | `pip install -r requirements.txt && pip install psycopg2-binary` | [ ] |
| 6 | Set real API keys in `.env` | Replace `YOUR_GEMINI_API_KEY` with actual Gemini API key | [ ] |
| 7 | Set real password in `db.py` | Replace `YOURUSERNAME` in `DATABASE_URL` with actual password | [ ] |
| 8 | Create database tables | `python -c "from models import Base; from db import engine; Base.metadata.create_all(engine)"` | [ ] |
| 9 | Verify embedding generation | Run verification script from Step 10 | [ ] |
| 10 | Verify vector store insert + search | Run verification script from Step 11 | [ ] |
| 11 | Seed vector store with 10+ stories | API seed / pipeline runs / lower threshold | [ ] |
| 12 | Create HNSW index | `CREATE INDEX ... USING hnsw (embedding vector_cosine_ops)` | [ ] |
| 13 | End-to-end verification | Two pipeline runs, check logs for retrieval messages | [ ] |
