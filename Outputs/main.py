from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import config
from agents import agent1_extraction
from agents import orchestrator

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- MODELS ---
class AnalyzeRequest(BaseModel):
    filename: str
    selected_pages: Optional[List[int]] = None  # None = analyze all extracted pages

# --- ENDPOINTS ---

@app.post("/upload-and-extract")
async def upload_and_extract(
    file: UploadFile = File(...),
    country: str = "AU"
):
    """
    Upload PDF and extract ALL pages with title detection.
    
    Args:
        file: PDF file upload
        country: Country code for title extraction (AU, IN, EU)
        
    Returns:
        Extraction results with title, total pages, and list of available pages
    """
    try:
        # Always extract all pages
        result = agent1_extraction.upload_and_extract_pdf(
            file.file,
            file.filename,
            selected_pages=None,  # None = all pages
            country=country
        )
        return {
            "status": "success",
            **result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    """
    Analyze extracted document using multi-agent pipeline.
    
    User can specify which pages to analyze. If not specified, all extracted pages are analyzed.
    
    Orchestrates:
    - Agent 2: Categorization with RAG
    - Agent 3: Domain and Department Assignment
    - Agent 4: Domain Expert Interpretations
    - Agent 5: Final Consolidated Interpretation
    
    Args:
        req: Analysis request with filename and optional selected_pages
        
    Returns:
        Complete analysis results with all agent outputs
    """
    try:
        result = orchestrator.orchestrate_analysis(
            filename=req.filename,
            selected_pages=req.selected_pages
        )
        return result
    except Exception as e:
        print(f"Analysis Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "config": {
            "llm_model": config.OLLAMA_MODEL,
            "llm_url": config.OLLAMA_URL,
            "chroma_db_path": str(config.CHROMA_DB_PATH),
            "temp_dir": str(config.TEMP_DATA_DIR)
        }
    }


if __name__ == "__main__":
    import uvicorn
    config.print_config_summary()
    uvicorn.run(app, host="0.0.0.0", port=8000)
