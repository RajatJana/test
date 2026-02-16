"""
Agent 2: Categorization with RAG
Categorizes regulatory text using LLM with RAG support.
Adds page numbers to output and leaves Domain/Responsible Dept. empty for Agent 3.
"""
import json
from typing import List, Dict, Any
from pathlib import Path
import config
from utils.rag_helper import get_rag_examples
from utils.json_helpers import safe_json_parse


def load_categorization_prompt() -> str:
    """Load the categorization prompt template."""
    prompt_path = config.PROMPTS_DIR / "categorization_prompt.txt"
    
    if not prompt_path.exists():
        raise FileNotFoundError(f"Categorization prompt not found: {prompt_path}")
    
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


def build_categorization_prompt(
    pages_data: List[Dict[str, Any]],
    rag_section: str
) -> str:
    """
    Build the complete categorization prompt with RAG examples.
    
    Args:
        pages_data: List of {page_number, text} dictionaries
        rag_section: Formatted RAG examples string
        
    Returns:
        Complete prompt string
    """
    base_prompt = load_categorization_prompt()
    
    # Concatenate page texts with page markers
    batch_content = ""
    for page_data in pages_data:
        page_num = page_data["page_number"]
        text = page_data["text"]
        batch_content += f"\n\n--- PAGE {page_num} ---\n\n{text}"
    
    # Build final prompt
    final_prompt = base_prompt
    
    if rag_section:
        final_prompt += "\n\n### 4.1 DYNAMIC RAG EXAMPLES (Reference)\n" + rag_section
    
    final_prompt += (
        "\n\n### 5. INPUT DATA\n"
        "Analyze the following text:\n"
        f"{batch_content}\n\n"
        "Output: JSON only.\n\n"
        "IMPORTANT: For each item, include a 'Page Number' field indicating which page the text was extracted from."
    )
    
    return final_prompt


def categorize_batch(
    pages_data: List[Dict[str, Any]],
    document_name: str
) -> Dict[str, Any]:
    """
    Categorize a batch of pages using LLM with RAG.
    
    Args:
        pages_data: List of {page_number, text} dictionaries
        document_name: Name of the document
        
    Returns:
        Dictionary containing:
            - data: List of categorized items with page numbers
            - trace: Processing trace for debugging
    """
    llm = config.get_llm_client()
    
    # Concatenate all page texts for RAG query
    full_batch_text = "\n\n".join([p["text"] for p in pages_data])
    
    # Get RAG examples
    rag_section, retrieved_items = get_rag_examples(full_batch_text)
    
    # Build prompt
    prompt = build_categorization_prompt(pages_data, rag_section)
    
    # Call LLM
    raw_response = llm.invoke(prompt, options={"temperature": config.LLM_TEMPERATURE})
    
    # Parse response
    try:
        data = safe_json_parse(raw_response)
    except Exception as e:
        return {
            "data": [],
            "trace": {
                "error": f"JSON parsing failed: {e}",
                "prompt": prompt[:500] + "...",
                "raw_response": raw_response[:1000] if raw_response else "",
                "rag_retrieved_items": retrieved_items
            }
        }
    
    # Ensure data is a list
    if not isinstance(data, list):
        data = [data] if isinstance(data, dict) else []
    
    # Post-process: Add document name and ensure page numbers
    for idx, item in enumerate(data):
        if isinstance(item, dict):
            item["Sr. No."] = idx + 1
            item["Document Name"] = document_name
            
            # Ensure page number exists (fallback to first page if missing)
            if "Page Number" not in item and pages_data:
                item["Page Number"] = pages_data[0]["page_number"]
            
            # Ensure Domain and Responsible Dept. are empty (filled by Agent 3)
            if "Domain" not in item:
                item["Domain"] = ""
            if "Responsible Dept." not in item:
                item["Responsible Dept."] = ""
    
    return {
        "data": data,
        "trace": {
            "rag_retrieved_items": retrieved_items,
            "prompt_preview": prompt[:300] + "...",
            "raw_response_preview": raw_response[:400] if raw_response else "",
            "items_count": len(data)
        }
    }


def categorize_all_pages(
    filename: str,
    document_name: str,
    selected_pages: List[int] = None
) -> Dict[str, Any]:
    """
    Categorize extracted pages with batch processing.
    
    Args:
        filename: Original PDF filename
        document_name: Document title
        selected_pages: Optional list of page numbers to analyze (None = all pages)
        
    Returns:
        Dictionary with all categorized items and traces
    """
    # Load extracted pages
    name_no_ext = filename.replace(".pdf", "")
    folder_name = f"{name_no_ext}_extracted"
    extract_dir = config.TEMP_DATA_DIR / folder_name
    
    if not extract_dir.exists():
        raise FileNotFoundError(f"Extracted pages not found: {extract_dir}")
    
    # Get all page files
    page_files = sorted(
        extract_dir.glob("page_*.md"),
        key=lambda f: int(f.stem.split("_")[1])
    )
    
    # Load page data
    all_pages_data = []
    for page_file in page_files:
        page_num = int(page_file.stem.split("_")[1])
        
        # Filter by selected_pages if specified
        if selected_pages is not None and page_num not in selected_pages:
            continue
        
        with open(page_file, "r", encoding="utf-8") as f:
            text = f.read()
        all_pages_data.append({"page_number": page_num, "text": text})
    
    # Batch processing
    batch_size = config.BATCH_SIZE_AGENT2
    all_results = []
    all_traces = []
    
    for i in range(0, len(all_pages_data), batch_size):
        batch = all_pages_data[i:i + batch_size]
        result = categorize_batch(batch, document_name)
        
        all_results.extend(result.get("data", []))
        all_traces.append({
            "batch_index": (i // batch_size) + 1,
            **result.get("trace", {})
        })
    
    # Re-number items
    for idx, item in enumerate(all_results):
        item["Sr. No."] = idx + 1
    
    return {
        "status": "success",
        "total_items": len(all_results),
        "data": all_results,
        "agent2_traces": all_traces
    }
