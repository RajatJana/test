"""
Orchestrator Agent
Coordinates the multi-agent workflow and manages state between agents.
"""
import json
from typing import Dict, Any
from pathlib import Path
import config
from agents import agent2_categorization
from agents import agent3_domain_assignment
from agents import agent4_expert_interpretation
from agents import agent5_consolidation


def orchestrate_analysis(
    filename: str, 
    selected_pages: List[int] = None
) -> Dict[str, Any]:
    """
    Main orchestration function that coordinates all 5 agents.
    
    Workflow:
    1. Load extracted page data from temp storage (created by Agent 1)
    2. Filter to selected pages (if specified)
    3. Agent 2: Categorization with RAG
    4. Agent 3: Domain and Department Assignment
    5. Agent 4: Domain Expert Interpretations
    6. Agent 5: Final Consolidated Interpretation
    
    Args:
        filename: Original PDF filename
        selected_pages: Optional list of page numbers to analyze (None = all pages)
        
    Returns:
        Dictionary with final results and processing traces
    """
    # Load metadata
    meta_path = config.TEMP_DATA_DIR / f"{filename}.meta.json"
    
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Metadata not found for {filename}. Please upload and extract first."
        )
    
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    document_name = metadata.get("document_name", "Unknown Document")
    
    print(f"\n{'='*60}")
    print(f"🚀 Starting Analysis Pipeline for: {document_name}")
    print(f"{'='*60}\n")
    
    # ==================== Agent 2: Categorization ====================
    if selected_pages:
        print(f"📋 Agent 2: Categorizing selected pages {selected_pages} with RAG...")
    else:
        print("📋 Agent 2: Categorizing all extracted pages with RAG...")
    
    try:
        agent2_result = agent2_categorization.categorize_all_pages(
            filename, 
            document_name,
            selected_pages=selected_pages
        )
        categorized_items = agent2_result.get("data", [])
        agent2_traces = agent2_result.get("agent2_traces", [])
        print(f"   ✅ Categorized {len(categorized_items)} items")
    except Exception as e:
        print(f"   ❌ Agent 2 failed: {e}")
        return {
            "status": "error",
            "agent": "agent2_categorization",
            "error": str(e),
            "data": []
        }
    
    # ==================== Agent 3: Domain Assignment ====================
    print("🏢 Agent 3: Assigning domains and departments...")
    try:
        agent3_result = agent3_domain_assignment.assign_domain_and_department(categorized_items)
        assigned_items = agent3_result.get("data", [])
        agent3_traces = agent3_result.get("agent3_traces", [])
        print(f"   ✅ Assigned domains and departments")
    except Exception as e:
        print(f"   ❌ Agent 3 failed: {e}")
        return {
            "status": "error",
            "agent": "agent3_domain_assignment",
            "error": str(e),
            "data": categorized_items,
            "agent2_traces": agent2_traces
        }
    
    # ==================== Agent 4: Expert Interpretations ====================
    print("👨‍💼 Agent 4: Adding domain expert interpretations...")
    try:
        agent4_result = agent4_expert_interpretation.add_domain_expert_interpretations(assigned_items)
        interpreted_items = agent4_result.get("data", [])
        agent4_traces = agent4_result.get("agent4_traces", [])
        print(f"   ✅ Added expert interpretations")
    except Exception as e:
        print(f"   ❌ Agent 4 failed: {e}")
        return {
            "status": "error",
            "agent": "agent4_expert_interpretation",
            "error": str(e),
            "data": assigned_items,
            "agent2_traces": agent2_traces,
            "agent3_traces": agent3_traces
        }
    
    # ==================== Agent 5: Final Consolidation ====================
    print("🎯 Agent 5: Generating final consolidated interpretations...")
    try:
        agent5_result = agent5_consolidation.add_final_consolidated_interpretation(interpreted_items)
        final_items = agent5_result.get("data", [])
        agent5_traces = agent5_result.get("agent5_traces", [])
        print(f"   ✅ Generated final interpretations")
    except Exception as e:
        print(f"   ❌ Agent 5 failed: {e}")
        return {
            "status": "error",
            "agent": "agent5_consolidation",
            "error": str(e),
            "data": interpreted_items,
            "agent2_traces": agent2_traces,
            "agent3_traces": agent3_traces,
            "agent4_traces": agent4_traces
        }
    
    # ==================== Save Final Output ====================
    name_no_ext = filename.replace(".pdf", "")
    folder_name = f"{name_no_ext}_extracted"
    output_dir = config.TEMP_DATA_DIR / folder_name
    
    final_json_path = output_dir / "final_output.json"
    with open(final_json_path, "w", encoding="utf-8") as f:
        json.dump(final_items, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"✅ Analysis Complete!")
    print(f"   Total Items: {len(final_items)}")
    print(f"   Output Saved: {final_json_path}")
    print(f"{'='*60}\n")
    
    # ==================== Return Complete Results ====================
    return {
        "status": "success",
        "document_name": document_name,
        "total_items": len(final_items),
        "data": final_items,
        "processing_traces": {
            "agent2_categorization": agent2_traces,
            "agent3_domain_assignment": agent3_traces,
            "agent4_expert_interpretation": agent4_traces,
            "agent5_consolidation": agent5_traces
        },
        "output_file": str(final_json_path)
    }
