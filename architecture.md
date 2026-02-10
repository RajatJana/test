# ClarifAI - Architecture

## 1. Overview

ClarifAI is an AI-powered requirement analysis and automation platform. It takes unstructured Business Requirement Documents (BRDs) as input and produces structured, actionable artifacts: classified requirements, user stories, BDD Gherkin scenarios, test cases, and a full traceability matrix.

The core engine is a sequential pipeline of specialized agents, each powered by Google Gemini (LLM). The application is accessible through three interfaces: a Streamlit web UI, a FastAPI REST API, and a CLI script.

---

## 2. High-Level Architecture

```
                         +-----------------+
                         |   Input Layer   |
                         +-----------------+
                        /        |          \
              +---------+  +-----------+  +----------+
              |   CLI   |  | Streamlit |  | FastAPI  |
              | main.py |  |    UI     |  | REST API |
              +---------+  +-----------+  +----------+
                        \        |          /
                         v       v        v
                    +-------------------------+
                    |     Agent Pipeline      |
                    |  (Sequential Stages)    |
                    +-------------------------+
                    | 1. ExtractorAgent       |
                    | 2. ClassifierAgent      |
                    | 3. ValidationAgent      |
                    | 4. UserStoryAgent       |
                    | 5. GherkinAgent         |
                    | 6. TestCaseAgent        |
                    | 7. TraceabilityAgent    |
                    +-------------------------+
                              |
                    +-------------------+
                    |   LLM Service     |
                    | (Google Gemini    |
                    |  2.5 Flash)       |
                    +-------------------+
                              |
                    +-------------------------+
                    |     Output Layer        |
                    +-------------------------+
                    |  Jira  | Excel | JSON   |
                    +-------------------------+
```

---

## 3. Project Structure

```
ClarifAI/
└── ClarifAI/                        # Main application package
    ├── main.py                      # CLI entry point - runs full pipeline
    ├── clarifAI_streamlit.py        # Streamlit tabbed web UI
    ├── clarifAI_wizard.py           # Streamlit step-by-step wizard UI
    ├── models.py                    # SQLAlchemy ORM models (ProgressTracker)
    ├── db.py                        # PostgreSQL database configuration
    ├── .env                         # Environment variables (API keys)
    ├── requirements.txt             # Python dependencies
    │
    ├── agents/                      # AI agents (one per pipeline stage)
    │   ├── extractor_agent.py       # Stage 1: Extract requirements from docs
    │   ├── classifier_agent.py      # Stage 2: Classify requirements
    │   ├── validation_agent.py      # Stage 3: Validate requirement quality
    │   ├── user_story_agent.py      # Stage 4: Generate user stories
    │   ├── gherkin_agent.py         # Stage 5: Generate BDD scenarios
    │   ├── testcase_agent.py        # Stage 6: Generate test cases
    │   ├── traceability_agent.py    # Stage 7: Build traceability matrix
    │   ├── new_traceability_agent.py        # Simplified traceability (US-based)
    │   ├── image_extractor_agent.py         # Extract info from UI screenshots
    │   ├── requirement_user_story_agent.py  # Combined extraction + story gen
    │   ├── user_story_feedback_agent.py     # Update stories from feedback
    │   └── acceptance_criteria_update_agent.py  # Regenerate acceptance criteria
    │
    ├── api/                         # FastAPI REST API layer
    │   ├── main.py                  # FastAPI app setup with CORS
    │   └── endpoints/
    │       └── requirements.py      # All API route handlers
    │
    ├── tools/                       # Shared tool functions
    │   ├── llm_utils.py             # Gemini API caller (call_gemini)
    │   ├── extractor_tools.py       # Document loading & requirement extraction
    │   ├── classifier_tools.py      # Requirement classification logic
    │   ├── prompt_utils.py          # Prompt building & JSON parsing
    │   └── doc_parsers.py           # PDF/DOCX file parsers
    │
    ├── utils/                       # Integration & export utilities
    │   ├── jira_utils.py            # Jira issue creation/update
    │   ├── export_us.py             # Export user stories to Excel
    │   ├── export_tc.py             # Export test cases to Excel
    │   ├── logger.py                # Rotating file logger setup
    │   └── pdf_utils.py             # PDF text extraction
    │
    ├── logs/                        # Log output directory
    └── sample_docs/                 # Sample BRD documents for testing
```

---

## 4. Entry Points

### 4.1 CLI (`ClarifAI/main.py`)

The CLI entry point runs the full agent pipeline sequentially. It implements API key rotation via `itertools.cycle()` to alternate between two Gemini API keys across LLM calls.

**Flow:**
1. Load document path (hardcoded or configurable)
2. Run `ExtractorAgent` -> extracted requirements
3. Run `ClassifierAgent` -> classified requirements
4. Run `ValidationAgent` -> validated requirements
5. Run `UserStoryAgent` -> user stories
6. Post user stories to Jira (if any)
7. Run `GherkinAgent` -> Gherkin scenarios
8. Run `TestCaseAgent` -> test cases
9. Post test cases to Jira (if any)

### 4.2 Streamlit Web UI (`ClarifAI/clarifAI_streamlit.py`)

An interactive tabbed interface with 8 tabs, each corresponding to a pipeline stage. Users upload a `.docx` or `.pdf` file, then progress through the tabs sequentially. Results are stored in Streamlit session state and passed between tabs.

**Tabs:**
1. **Extractor** - Upload BRD, extract requirements
2. **Classifier** - Classify extracted requirements (with confidence threshold slider)
3. **Validator** - Validate classified requirements
4. **User Stories** - Generate user stories (with confidence filter + Jira push)
5. **Gherkin** - Generate BDD scenarios (with `.feature` file download)
6. **Test Cases** - Generate test cases
7. **Traceability** - Build and display interactive traceability matrix (AgGrid)
8. **Logs** - View application execution logs in real-time

### 4.3 Streamlit Wizard (`ClarifAI/clarifAI_wizard.py`)

A step-by-step guided version of the Streamlit UI. Presents 6 sequential steps with a progress indicator and forward/back navigation. Uses the same agents and session state management.

### 4.4 FastAPI REST API (`ClarifAI/api/main.py`)

A programmatic REST API that exposes each pipeline stage as an independent POST endpoint. Uses CORS middleware (allowing all origins). See Section 7 for the full endpoint list.

---

## 5. Agent Pipeline: Input to Output

This is the core processing flow. Each stage receives the output of the previous stage, calls the Gemini LLM with a structured prompt, parses the JSON response, and passes results forward.

### Stage 1: Extraction

**Agent:** `ExtractorAgent` (`agents/extractor_agent.py`)
**Tools used:** `load_document_tool()`, `extract_requirements_tool()` from `tools/extractor_tools.py`

**Flow:**
1. Receives a file path (`.docx` or `.pdf`)
2. `load_document_tool()` detects the file type and delegates to the appropriate parser:
   - `.docx` -> `load_docx_tool()` (uses `python-docx`, adds line numbers)
   - `.pdf` -> `load_pdf_tool()` (uses `pdfplumber`, preserves page/line numbers)
3. The raw text (with line numbers) is passed to `extract_requirements_tool()`
4. A detailed prompt instructs Gemini to identify individual requirements from the raw text
5. Gemini returns structured JSON with: `requirement_id`, `requirement_text`, `original_text`, `line_number`, `page_number`, `source_section`
6. The JSON is parsed via `extract_clean_json()` (strips markdown fences if present)

**Output:** List of extracted requirements

### Stage 2: Classification

**Agent:** `ClassifierAgent` (`agents/classifier_agent.py`)
**Tools used:** `classify_requirement_tool()` from `tools/classifier_tools.py`

**Flow:**
1. Receives the list of extracted requirements
2. For each requirement, sends a prompt to Gemini requesting classification
3. Gemini classifies each requirement as:
   - **Functional** (subcategories: Business Rules, User Interactions, Authentication/Authorization, Data Management, System, External Integration, Reporting, Transaction Handling, Compliance, Core)
   - **Non-Functional** (subcategories: Security, Performance, Usability, Reliability, Availability, Maintainability, Monitoring, Scalability, Compliance)
4. Each classification includes a `confidence_score` (0.0-1.0) and identified `stakeholders`
5. Results are merged back into the original requirement objects

**Output:** Classified requirements (original fields + `requirement_type`, `sub_category`, `confidence_score`, `stakeholders`)

### Stage 3: Validation

**Agent:** `ValidationAgent` (`agents/validation_agent.py`)

**Flow:**
1. Receives classified requirements
2. Formats all requirements into a single batch prompt
3. Gemini evaluates each requirement against 4 quality criteria:
   - **Measurable and Verifiable** - includes specific metrics (time, percentage, numbers)
   - **Unambiguous and Clear** - no vague terms ("fast", "robust", "user-friendly")
   - **Atomic** - single, self-contained need (no compound statements)
   - **Complete** - answers what, who, when, where, how
4. Gemini returns per-requirement: `llm_check_passed` (boolean), `issues` (array of strings), `justification` (explanation)
5. Results are merged into each requirement under a `validation` key

**Output:** Validated requirements (classified fields + `validation` object)

### Stage 4: User Story Generation

**Agent:** `UserStoryAgent` (`agents/user_story_agent.py`)

**Flow:**
1. Receives validated requirements
2. **Filters** to only process requirements where:
   - `requirement_type` == "Functional"
   - `validation.llm_check_passed` == true
3. Sends the filtered list to Gemini with a product-owner persona prompt
4. Gemini generates one or more user stories per requirement, each containing:
   - `user_story_id` (auto-incrementing: US-001, US-002, ...)
   - `title`, `user_story` ("As a ... I want ... so that ..."), `description`
   - `acceptance_criteria` (3-4 items)
   - `confidence_score`, `tshirt_size` (XS/S/M/L/XL), `priority` (High/Medium/Low), `tags`
5. Response is parsed from JSON

**Output:** List of user stories

### Stage 5: Gherkin BDD Scenario Generation

**Agent:** `GherkinAgent` (`agents/gherkin_agent.py`)

**Flow:**
1. Receives user stories
2. Prompts Gemini to generate BDD test scenarios in Gherkin format
3. Each scenario includes: `feature` name, `scenarios` (with `name` and `steps` in Given/When/Then format), `confidence_score`
4. Scenarios are linked back to requirements via `requirement_id`

**Output:** List of Gherkin scenarios

### Stage 6: Test Case Generation

**Agent:** `TestCaseAgent` (`agents/testcase_agent.py`)

**Flow:**
1. Receives user stories
2. Prompts Gemini to generate comprehensive test cases covering:
   - Positive scenarios (happy path)
   - Negative scenarios (error handling, invalid inputs)
   - Boundary value analysis
   - Optional field scenarios
   - Combinatorial and priority-based scenarios
3. Each test case includes: `test_id` (format: `TC-REQ-001-US-001-001`), `title`, `precondition`, `steps`, `expected_result`, `priority`, `tags`

**Output:** List of test cases

### Stage 7: Traceability Matrix

**Agent:** `TraceabilityAgent` (`agents/traceability_agent.py`)

**Flow:**
1. Receives all previous stage outputs (classified, validated, user stories, gherkin, test cases)
2. Collects all unique `requirement_id` values across all stages
3. Deduplicates user stories by `(requirement_id, user_story_id)` pairs
4. Parses test case IDs to extract embedded `requirement_id` and `user_story_id`
5. For each requirement, builds a trace entry linking:
   - Requirement text and type
   - Classification confidence
   - Validation status and issues
   - Associated user stories with acceptance criteria
   - Gherkin feature and scenarios
   - Test cases

**Output:** Full traceability matrix (list of trace entries, one per requirement)

---

## 6. Supporting Agents

Beyond the core pipeline, additional agents handle specialized workflows:

| Agent | File | Purpose |
|-------|------|---------|
| `ImageExtractorAgent` | `agents/image_extractor_agent.py` | Extracts structured info from UI/UX wireframe images (SVG, PNG, JPG) using Gemini Vision |
| `RequirementUserStoryAgent` | `agents/requirement_user_story_agent.py` | Combined flow: extract from images + optional reference docs -> user stories directly |
| `UserStoryFeedbackAgent` | `agents/user_story_feedback_agent.py` | Updates a user story based on textual feedback |
| `AcceptanceCriteriaUpdateAgent` | `agents/acceptance_criteria_update_agent.py` | Regenerates acceptance criteria from a description and story |
| `NewTraceabilityAgent` | `agents/new_traceability_agent.py` | Simplified traceability: user stories -> gherkin -> test cases (skips requirement-level) |

---

## 7. Tools & Utilities

### LLM Service (`tools/llm_utils.py`)

- **Function:** `call_gemini(prompt, api_key, model)`
- **Model:** `gemini-2.5-flash`
- **Configuration:** Temperature = 0.0 (deterministic output)
- Supports optional API key injection for key rotation

### Document Parsing

| Tool | File | Description |
|------|------|-------------|
| `load_document_tool()` | `tools/extractor_tools.py` | Auto-detects file type and routes to correct parser |
| `load_docx_tool()` | `tools/extractor_tools.py` | Parses `.docx` files with line numbers |
| `load_pdf_tool()` | `tools/extractor_tools.py` | Parses `.pdf` files with page and line numbers via `pdfplumber` |
| `extract_requirements_tool()` | `tools/extractor_tools.py` | Sends parsed text to Gemini for requirement extraction |
| `extract_text_from_file()` | `tools/doc_parsers.py` | Alternative parser supporting PDF (PyMuPDF) and DOCX |
| `extract_tables_from_file()` | `tools/doc_parsers.py` | Extracts tabular data from documents |

### Prompt & JSON Utilities (`tools/prompt_utils.py`)

- `build_json_prompt()` - Constructs prompts that request JSON output from Gemini
- `extract_clean_json()` - Strips markdown fences and parses JSON from LLM responses

### Classification (`tools/classifier_tools.py`)

- `classify_requirement_tool()` - Sends batch of requirements to Gemini for classification, returns enriched requirement objects

### Export Utilities

| Utility | File | Description |
|---------|------|-------------|
| `export_user_stories_to_excel()` | `utils/export_us.py` | Flattens user story JSON and writes to `.xlsx` |
| `export_test_cases_to_excel()` | `utils/export_tc.py` | Flattens test case JSON and writes to `.xlsx` |

### Logging (`utils/logger.py`)

- Rotating file handler: 1 MB max file size, 3 backup files
- Writes to `logs/ClarifAI.log`
- Console + file output with timestamped formatting

---

## 8. API Endpoints

All endpoints are defined in `api/endpoints/requirements.py` and mounted via `FastAPI` in `api/main.py`.

### Core Pipeline Endpoints

| Method | Path | Input | Description |
|--------|------|-------|-------------|
| POST | `/extract` | File upload (`.docx`/`.pdf`) | Extract requirements from BRD document |
| POST | `/classify` | JSON body (extracted requirements) | Classify requirements as Functional/Non-Functional |
| POST | `/validate` | JSON body (classified requirements) | Validate requirement quality |
| POST | `/user_stories` | JSON body (validated requirements) | Generate user stories |
| POST | `/gherkin` | JSON body (user stories) | Generate BDD Gherkin scenarios |
| POST | `/test_cases` | JSON body (user stories) | Generate test cases |
| POST | `/traceability` | JSON body (all stage outputs) | Build full traceability matrix |
| POST | `/newtraceability` | JSON body (user stories + optional gherkin/test cases) | Build simplified traceability |

### Image & Combined Endpoints

| Method | Path | Input | Description |
|--------|------|-------|-------------|
| POST | `/extract-info` | Multiple image files | Extract structured info from UI/UX wireframes |
| POST | `/generate-userstory` | Form data (raw_info + optional reference files) | Generate user stories from extracted image info |
| POST | `/extract-and-generate` | Image files + optional reference files | Combined: extract from images then generate user stories |

### Feedback & Update Endpoints

| Method | Path | Input | Description |
|--------|------|-------|-------------|
| POST | `/update-story` | JSON body (story + feedback text) | Update user story based on feedback |
| POST | `/update-acceptance-criteria` | JSON body (description + story) | Regenerate acceptance criteria |

### Export & Integration Endpoints

| Method | Path | Input | Description |
|--------|------|-------|-------------|
| POST | `/export-userstories` | JSON body (user stories) | Download user stories as Excel file |
| POST | `/export-testcases` | JSON body (test cases) | Download test cases as Excel file |
| POST | `/jira/user_stories` | JSON body (user stories) | Create/update user stories in Jira |
| POST | `/jira/test_cases` | JSON body (test cases) | Create/update test cases in Jira |
| POST | `/convert-yaml-to-requirements-json` | YAML file upload | Convert YAML requirements to JSON format |

---

## 9. External Integrations

### Google Gemini (LLM)

- **Model:** `gemini-2.5-flash`
- **Purpose:** Powers all AI agents (extraction, classification, validation, generation)
- **Configuration:** API keys via `.env` (`GOOGLE_API_KEY_1`, `GOOGLE_API_KEY_2`)
- **Key rotation:** CLI uses `itertools.cycle()` to alternate keys across calls

### Jira

- **Purpose:** Syncs generated user stories and test cases as Jira issues
- **Operations:** Bulk search, bulk create, individual update
- **Auth:** HTTP Basic Auth (email + API token)
- **Configuration:** Hardcoded in `utils/jira_utils.py` (URL, email, token, project key)

### PostgreSQL

- **Purpose:** Tracks document processing progress
- **ORM:** SQLAlchemy
- **Table:** `clarifai_progress_tracker` (doc_id, doc_name, stage, output_paths, date_last_modified)
- **Configuration:** Connection string in `db.py`

---

## 10. Configuration

### Environment Variables (`.env`)

| Variable | Purpose |
|----------|---------|
| `GOOGLE_API_KEY_1` | Primary Gemini API key |
| `GOOGLE_API_KEY_2` | Secondary Gemini API key (for rotation) |
| `GOOGLE_BASE_URL` | Gemini API endpoint |
| `GEMINI_API_KEY` | Alternative API key reference |

### Running the Application

```bash
# FastAPI REST API
uvicorn api.main:app --reload
uvicorn api.main:app --timeout-keep-alive 120

# Streamlit Web UI
streamlit run clarifAI_streamlit.py

# Streamlit Wizard
streamlit run clarifAI_wizard.py

# CLI Pipeline
python main.py
```

### Dependencies

Python packages (from `requirements.txt`):
- **AI/LLM:** `google-generativeai`, `google-adk`
- **Web UI:** `streamlit`, `streamlit-aggrid`
- **REST API:** `fastapi`, `uvicorn`, `pydantic`
- **Document parsing:** `python-docx`, `pdfplumber`
- **Data processing:** `pandas`, `openpyxl`
- **Visualization:** `matplotlib`, `seaborn`
- **HTTP:** `requests`
