"""
Agent 3: Domain and Department Assignment
Assigns domain and responsible department using LLM with full department context.
"""
import json
from typing import List, Dict, Any
from pathlib import Path
import config
from utils.json_helpers import safe_json_parse


def load_departments() -> List[Dict[str, Any]]:
    """Load department data from JSON file."""
    if not config.DEPARTMENTS_JSON.exists():
        print(f"⚠️ Departments file not found: {config.DEPARTMENTS_JSON}")
        return []
    
    with open(config.DEPARTMENTS_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    return data if isinstance(data, list) else []


def build_assignment_prompt(items: List[Dict[str, Any]], departments: List[Dict[str, Any]]) -> str:
    """
    Build prompt for domain and department assignment.
    
    Args:
        items: Categorized items from Agent 2
        departments: Full department list with all details
        
    Returns:
        Complete prompt string
    """
    # Extract department names for the constraint
    dept_names = [d.get("departmentName", "").strip() for d in departments if d.get("departmentName")]
    dept_names = [d for d in dept_names if d]  # Remove empty
    
    # Format full department information for context
    dept_context = []
    for dept in departments:
        dept_name = dept.get("departmentName", "")
        activity = dept.get("ActivityDesc", "")
        products = dept.get("ProdServc", "")
        regulatory = dept.get("RegulatoryFocus", "")
        
        if dept_name:
            dept_info = f"""
**{dept_name}**
- Activities: {activity}
- Products/Services: {products}
- Regulatory Focus: {regulatory}
"""
            dept_context.append(dept_info.strip())
    
    dept_context_str = "\n\n".join(dept_context)
    
    prompt = f"""You are a banking domain expert. Your task is to assign the appropriate Domain and Responsible Department for each regulatory requirement.

**DOMAIN OPTIONS (choose ONE per item):**
- NA
- Retail Banking
- Mortgage & Retail Loans
- Commercial Loans
- Cards
- Wealth Mgmt. & Investments
- Agri-Banking
- Payments

**AVAILABLE DEPARTMENTS:**
Below is the complete list of departments with their activities, products/services, and regulatory focus.
Use this information to understand what each department does and assign the most appropriate one.

{dept_context_str}

**IMPORTANT RULES:**
1. The "Responsible Dept." MUST be EXACTLY one of these department names: {json.dumps(dept_names, ensure_ascii=False)}
2. Do NOT invent new department names
3. If you cannot determine confidently, set "Responsible Dept." to null
4. Choose the Domain that best matches the regulatory requirement
5. Consider the department's activities, products, and regulatory focus when assigning
6. Return ONLY valid JSON (no markdown fences, no explanations)

**INPUT ITEMS:**
{json.dumps(items, ensure_ascii=False, indent=2)}

**OUTPUT FORMAT:**
Return a JSON array where each object has:
{{
  "Sr. No.": <same as input>,
  "Domain": "<one of the domain options>",
  "Responsible Dept.": "<exactly one department name from the list or null>"
}}

Output JSON only:
"""
    
    return prompt


def assign_domain_and_department_batch(
    items: List[Dict[str, Any]],
    departments: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Assign domain and department for a batch of items.
    
    Args:
        items: Categorized items from Agent 2
        departments: Full department list
        
    Returns:
        Dictionary with updated items and trace
    """
    if not items:
        return {"data": [], "trace": {}}
    
    llm = config.get_llm_client()
    
    # Build prompt
    prompt = build_assignment_prompt(items, departments)
    
    # Call LLM
    raw_response = llm.invoke(prompt, options={"temperature": config.LLM_TEMPERATURE})
    
    # Parse response
    try:
        assignments = safe_json_parse(raw_response)
    except Exception as e:
        return {
            "data": items,  # Return original items if parsing fails
            "trace": {
                "error": f"JSON parsing failed: {e}",
                "prompt_preview": prompt[:500] + "...",
                "raw_response": raw_response[:1000] if raw_response else ""
            }
        }
    
    # Ensure assignments is a list
    if not isinstance(assignments, list):
        assignments = [assignments] if isinstance(assignments, dict) else []
    
    # Build lookup by Sr. No.
    assignments_by_srno = {}
    for assignment in assignments:
        if isinstance(assignment, dict) and "Sr. No." in assignment:
            assignments_by_srno[assignment["Sr. No."]] = assignment
    
    # Merge assignments back into items
    for item in items:
        sr_no = item.get("Sr. No.")
        if sr_no in assignments_by_srno:
            assignment = assignments_by_srno[sr_no]
            item["Domain"] = assignment.get("Domain", "")
            item["Responsible Dept."] = assignment.get("Responsible Dept.", "")
    
    return {
        "data": items,
        "trace": {
            "prompt_preview": prompt[:300] + "...",
            "raw_response_preview": raw_response[:400] if raw_response else "",
            "assignments_count": len(assignments)
        }
    }


def assign_domain_and_department(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Assign domain and department for all items with batch processing.
    
    Args:
        items: Categorized items from Agent 2
        
    Returns:
        Dictionary with updated items and traces
    """
    departments = load_departments()
    
    if not departments:
        print("⚠️ No departments loaded. Skipping domain/department assignment.")
        return {
            "status": "warning",
            "message": "No departments available",
            "data": items,
            "agent3_traces": []
        }
    
    batch_size = config.BATCH_SIZE_AGENT3
    all_traces = []
    
    # Process in batches
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        result = assign_domain_and_department_batch(batch, departments)
        
        # Update items in place (result["data"] contains updated batch)
        items[i:i + batch_size] = result["data"]
        
        all_traces.append({
            "batch_index": (i // batch_size) + 1,
            **result.get("trace", {})
        })
    
    return {
        "status": "success",
        "data": items,
        "agent3_traces": all_traces
    }
