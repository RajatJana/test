# RegAssist - Multi-Agent Regulatory Analysis System

RegAssist is a modular, multi-agent system for processing and analyzing regulatory documents. It uses a pipeline of specialized agents to extract, categorize, and interpret regulatory requirements.

## Architecture

### 5 Specialized Agents

1. **Agent 1: Text Extraction & Title Detection**
   - Extracts text page-wise from PDF documents
   - Detects document title using rule-based approach (no LLM)
   - Supports country-specific profiles (AU, IN, EU)

2. **Agent 2: Categorization with RAG**
   - Categorizes regulatory text using LLM
   - Retrieves similar examples from ChromaDB (RAG)
   - Tracks page numbers throughout processing

3. **Agent 3: Domain & Department Assignment**
   - Assigns banking domain (Retail, Commercial Loans, Cards, etc.)
   - Assigns responsible department from predefined list
   - Uses full department context for accurate assignment

4. **Agent 4: Domain Expert Interpretations**
   - Adds domain-specific expert interpretations
   - Adds 4 fixed expert interpretations (Legal, Risk, Geo-specific, Compliance)
   - Provides regulatory-aware, practical guidance

5. **Agent 5: Final Consolidated Interpretation**
   - Synthesizes all expert interpretations
   - Generates final, implementation-ready guidance
   - Resolves conflicts conservatively

### Orchestrator

The orchestrator coordinates all agents, manages state, handles errors, and provides progress tracking.

## Project Structure

```
RegAssist/new/
├── agents/
│   ├── agent1_extraction.py          # Text extraction & title detection
│   ├── agent2_categorization.py      # Categorization with RAG
│   ├── agent3_domain_assignment.py   # Domain & department assignment
│   ├── agent4_expert_interpretation.py # Expert interpretations
│   ├── agent5_consolidation.py       # Final consolidation
│   └── orchestrator.py               # Multi-agent coordinator
├── utils/
│   ├── country_profiles.py           # Country-specific title extraction rules
│   ├── title_extractor.py            # Rule-based title extraction
│   ├── rag_helper.py                 # ChromaDB operations
│   └── json_helpers.py               # JSON parsing utilities
├── prompts/
│   └── categorization_prompt.txt     # Base categorization prompt
├── config.py                         # Centralized configuration
├── main.py                           # FastAPI endpoints
├── bank_departments_europe.json      # Department definitions
├── .env.example                      # Environment variables template
└── requirements.txt                  # Python dependencies
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Copy `.env.example` to `.env` and update values:

```bash
cp .env.example .env
```

Key configuration:
- `OLLAMA_URL`: Your LLM service endpoint
- `OLLAMA_MODEL`: Model identifier (e.g., gemma3:12b)
- `CHROMA_DB_PATH`: Path to ChromaDB database
- `BATCH_SIZE_AGENT2-5`: Batch sizes for each agent

### 3. Ensure ChromaDB is Available

Make sure your ChromaDB database is accessible at the configured path.

## Usage

### Recommended Workflow

1. **Upload PDF** → System extracts ALL pages automatically
2. **Review extracted pages** → User decides which pages to analyze
3. **Analyze** → System processes selected pages (or all if not specified)

### Start the Server

```bash
python main.py
```

Or with uvicorn:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Selective Page Analysis

```bash
# Analyze only specific pages
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{"filename": "sample.pdf", "selected_pages": [1, 2, 3, 5, 10]}'

# Analyze all extracted pages
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{"filename": "sample.pdf"}'
```

### 3. Different Countries

```bash
# Upload a PDF and specify the country for title extraction
curl -X POST "http://localhost:8000/upload-and-extract?country=IN" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/your/document.pdf"
```

### API Endpoints

#### 1. Upload and Extract

```bash
POST /upload-and-extract
```

**Parameters:**
- `file`: PDF file (multipart/form-data)
- `country`: Country code for title extraction (default: "AU")

**Behavior:**
- Extracts **ALL pages** from the PDF automatically
- Detects document title using country-specific rules
- Saves extracted pages to temp storage

**Response:**
```json
{
  "status": "success",
  "filename": "document.pdf",
  "title": "Extracted Document Title",
  "total_pages": 50,
  "extracted_pages": 50,
  "output_dir": "path/to/extracted",
  "files": ["page_1.md", "page_2.md", ...]
}
```

#### 2. Analyze

```bash
POST /analyze
```

**Body:**
```json
{
  "filename": "document.pdf",
  "selected_pages": [1, 2, 5, 10, 15]  // Optional: specific pages to analyze
}
```

**Behavior:**
- If `selected_pages` is provided: analyzes only those pages
- If `selected_pages` is omitted: analyzes all extracted pages

**Response:**
```json
{
  "status": "success",
  "document_name": "Document Title",
  "total_items": 150,
  "data": [
    {
      "Sr. No.": 1,
      "Document Name": "...",
      "Page Number": 5,
      "Section": "...",
      "Sub-section": "...",
      "Text": "...",
      "Classification": "Action Point",
      "Responsible Dept.": "Risk Management",
      "Domain": "Commercial Loans",
      "Interpretation/Simplification": "...",
      "Commercial Loans Expert Interpretation/Simplification": "...",
      "Banking Legal Expert Interpretation/Simplification": "...",
      "Risk Expert Interpretation/Simplification": "...",
      "Geo-specific Financial Practice Expert Interpretation/Simplification": "...",
      "Compliance Expert Interpretation/Simplification": "...",
      "Final Interpretation/Simplification": "..."
    }
  ],
  "processing_traces": {
    "agent2_categorization": [...],
    "agent3_domain_assignment": [...],
    "agent4_expert_interpretation": [...],
    "agent5_consolidation": [...]
  }
}
```

#### 3. Health Check

```bash
GET /health
```

## Features

### ✅ Modular Architecture
- Each agent is independent and focused on a single responsibility
- Easy to modify, test, and extend individual agents

### ✅ Batch Processing
- All agents support batch processing for large documents
- Configurable batch sizes per agent

### ✅ Page Number Tracking
- Page numbers are preserved throughout the pipeline
- Final output includes source page for each item

### ✅ Rule-Based Title Extraction
- No LLM calls for title detection
- Fast and reliable using country-specific profiles

### ✅ RAG Integration
- Retrieves similar examples from ChromaDB
- Improves categorization accuracy

### ✅ Comprehensive Expert Analysis
- Domain-specific interpretations
- 4 fixed expert perspectives (Legal, Risk, Geo, Compliance)
- Final consolidated interpretation

### ✅ Centralized Configuration
- All settings in one place
- Environment variable support
- Easy to switch LLM providers

## Country Profiles

Supported countries for title extraction:
- **AU** (Australia): APRA standards (CPS, APS, LPS, GPS, etc.)
- **IN** (India): RBI circulars, master directions
- **EU** (Europe): EBA guidelines, regulations, directives

## Customization

### Adding New Country Profiles

Edit `utils/country_profiles.py`:

```python
COUNTRY_PROFILES["US"] = {
    "title_prefixes": ["Federal Regulation", "Banking Rule"],
    "junk_terms": ["federal register", "cfr"],
    "require_code": False
}
```

### Modifying Prompts

Edit files in `prompts/` directory to customize agent behavior.

### Adjusting Batch Sizes

Update `.env` file:

```
BATCH_SIZE_AGENT2=10
BATCH_SIZE_AGENT3=20
BATCH_SIZE_AGENT4=15
BATCH_SIZE_AGENT5=15
```

## Troubleshooting

### ChromaDB Connection Issues

Ensure ChromaDB path is correct in `.env`:
```
CHROMA_DB_PATH=../chroma_db
```

### LLM Connection Issues

Check Ollama is running and accessible:
```
OLLAMA_URL=http://localhost:11434
```

### Memory Issues

Reduce batch sizes in `.env` if running out of memory.

## License

[Your License Here]
