from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
from fastapi.responses import FileResponse
from fastapi import UploadFile, File, Form
import os
import shutil
import uuid
import tempfile
from agents.extractor_agent import ExtractorAgent
from agents.image_extractor_agent import ImageExtractorAgent
from agents.requirement_user_story_agent import RequirementUserStoryAgent
from agents.classifier_agent import ClassifierAgent
from agents.validation_agent import ValidationAgent
from agents.user_story_agent import UserStoryAgent
from agents.user_story_feedback_agent import UserStoryFeedbackAgent
from agents.gherkin_agent import GherkinAgent
from agents.testcase_agent import TestCaseAgent
from agents.traceability_agent import TraceabilityAgent
from agents.new_traceability_agent import NewTraceabilityAgent
from agents.acceptance_criteria_update_agent import AcceptanceCriteriaUpdateAgent
from tools.llm_utils import call_gemini
from utils.jira_utils import post_user_stories_to_jira, post_test_cases_to_jira
from utils.export_us import export_user_stories_to_excel
from utils.export_tc import export_test_cases_to_excel

router = APIRouter()


###########################################################################

# @router.post("/extract-info")
# async def extract_info(file: UploadFile = File(...)):
#     agent = ImageExtractorAgent()
#     try:
#         file_bytes = await file.read()
#         result = agent.run(file_bytes, file.filename)
#         return result  # {"raw_info": "..."}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
@router.post("/extract-info")
async def extract_info(files: List[UploadFile] = File(...)):
    agent = ImageExtractorAgent()
    combined_info = []
    errors = []
    
    for file in files:
        try:
            file_bytes = await file.read()
            result = agent.run(file_bytes, file.filename)
            extracted_info = result.get("raw_info", "")
            if not isinstance(extracted_info, str):
                extracted_info = str(extracted_info)  # Convert non-string to string
            combined_info.append(f"--- {file.filename} ---\n{extracted_info}")
        except Exception as e:
            errors.append(f"Error processing {file.filename}: {str(e)}")
    
    # Combine all extracted info into a single string
    final_output = "\n\n".join(combined_info)
    if errors:
        final_output += "\n\n--- Errors ---\n" + "\n".join(errors)
    
    return {"raw_info": final_output}
###########################################################################
UPLOAD_DIR = "/tmp/reference_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/generate-userstory")
async def generate_userstory(
    raw_info: str = Form(...),
    files: Optional[List[UploadFile]] = File(None)
):
    """
    Generate user stories from raw_info and optional reference documents.
    - raw_info: required text from UI/UX extraction
    - files: optional list of PDF/DOCX uploads
    """

    reference_paths = []
    if files:
        for f in files:
            try:
                ext = os.path.splitext(f.filename)[1]
                safe_name = f"{uuid.uuid4().hex}{ext}"
                file_path = os.path.join(UPLOAD_DIR, safe_name)

                with open(file_path, "wb") as buffer:
                    shutil.copyfileobj(f.file, buffer)

                reference_paths.append(file_path)
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to save file {f.filename}: {e}"
                )

    agent = RequirementUserStoryAgent(llm_caller=call_gemini)

    try:
        result = agent.run({
            "raw_info": raw_info,
            "reference_docs": reference_paths  # optional param
        })
        return result  # {"user_stories": [...]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
# class RawInfoRequest(BaseModel):
#     raw_info: str

# @router.post("/generate-userstory")
# async def generate_userstory(request: RawInfoRequest):
#     agent = RequirementUserStoryAgent(llm_caller=call_gemini)
#     try:
#         result = agent.run({"raw_info": request.raw_info})
#         return result  # {"user_stories": [...]}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
###########################################################################
# Single API for both extract-and-generate

# @router.post("/extract-and-generate")
# async def extract_and_generate(
#     image_file: UploadFile = File(..., description="UI/UX screenshot or image"),
#     reference_files: Optional[List[UploadFile]] = File(None, description="Optional reference docs (PDF/DOCX)")
# ):
#     """
#     1. Extract raw_info from image_file
#     2. Optionally process reference_files
#     3. Generate user stories
#     """

#     # Step 1: Extract info from image
#     agent_extractor = ImageExtractorAgent()
#     try:
#         image_bytes = await image_file.read()
#         raw_info_result = agent_extractor.run(image_bytes, image_file.filename)
#         raw_info = raw_info_result.get("raw_info") if isinstance(raw_info_result, dict) else raw_info_result

#         if not raw_info or not str(raw_info).strip():
#             raise HTTPException(status_code=400, detail="No raw_info extracted from image.")
#     except HTTPException:
#         raise
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Image extraction failed: {e}")

#     # Step 2: Save optional reference docs
#     reference_paths = []
#     if reference_files:
#         for f in reference_files:
#             try:
#                 ext = os.path.splitext(f.filename)[1]
#                 safe_name = f"{uuid.uuid4().hex}{ext}"
#                 file_path = os.path.join(UPLOAD_DIR, safe_name)

#                 with open(file_path, "wb") as buffer:
#                     shutil.copyfileobj(f.file, buffer)

#                 reference_paths.append(file_path)
#             except Exception as e:
#                 raise HTTPException(status_code=400, detail=f"Failed to save file {f.filename}: {e}")

#     # Step 3: Generate user stories
#     agent_userstory = RequirementUserStoryAgent(llm_caller=call_gemini)
#     try:
#         result = agent_userstory.run({
#             "raw_info": raw_info,
#             "reference_docs": reference_paths
#         })
#         return result  # {"user_stories": [...]}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"User story generation failed: {e}")
@router.post("/extract-and-generate")
async def extract_and_generate(
    image_files: List[UploadFile] = File(..., description="UI/UX screenshots or images"),
    reference_files: Optional[List[UploadFile]] = File(None, description="Optional reference docs (PDF/DOCX)")
):
    """
    1. Extract raw_info from multiple image_files and combine into a single output
    2. Optionally process reference_files
    3. Generate user stories based on combined raw_info and reference files
    """

    # Step 1: Extract info from images and combine
    agent_extractor = ImageExtractorAgent()
    combined_info = []
    errors = []

    for image_file in image_files:
        try:
            image_bytes = await image_file.read()
            raw_info_result = agent_extractor.run(image_bytes, image_file.filename)
            raw_info = raw_info_result.get("raw_info", "") if isinstance(raw_info_result, dict) else str(raw_info_result)
            
            if not raw_info or not str(raw_info).strip():
                errors.append(f"No raw_info extracted from {image_file.filename}")
                continue
                
            combined_info.append(f"--- {image_file.filename} ---\n{raw_info}")
        except Exception as e:
            errors.append(f"Error processing {image_file.filename}: {str(e)}")

    # Combine all extracted info into a single string
    final_raw_info = "\n\n".join(combined_info)
    if not final_raw_info and errors:
        raise HTTPException(status_code=400, detail="No valid information extracted from any image. Errors: " + "; ".join(errors))
    
    if errors:
        final_raw_info += "\n\n--- Errors ---\n" + "\n".join(errors)

    # Step 2: Save optional reference docs
    reference_paths = []
    if reference_files:
        try:
            for f in reference_files:
                ext = os.path.splitext(f.filename)[1]
                safe_name = f"{uuid.uuid4().hex}{ext}"
                file_path = os.path.join(UPLOAD_DIR, safe_name)

                with open(file_path, "wb") as buffer:
                    shutil.copyfileobj(f.file, buffer)

                reference_paths.append(file_path)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to save reference files: {str(e)}")

    # Step 3: Generate user stories
    agent_userstory = RequirementUserStoryAgent(llm_caller=call_gemini)
    try:
        result = agent_userstory.run({
            "raw_info": final_raw_info,
            "reference_docs": reference_paths
        })
        return result  # Expected to return {"user_stories": [...]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"User story generation failed: {str(e)}")    
##########################################################################
class AcceptanceCriteriaRequest(BaseModel):
    description: str
    story: str

@router.post("/update-acceptance-criteria")
async def update_acceptance_criteria(request: AcceptanceCriteriaRequest):
    """
    Generate acceptance criteria for a given user story and description.
    """
    agent = AcceptanceCriteriaUpdateAgent(llm_caller=call_gemini)
    result = agent.run(request.model_dump())

    if not result.get("acceptance_criteria"):
        raise HTTPException(
            status_code=500,
            detail="Failed to update acceptance criteria"
        )

    return result
###########################################################################

# class FilePathRequest(BaseModel):
#     file_path: str

# @router.post("/extract")
# async def extract_requirements(request: FilePathRequest):
#     agent = ExtractorAgent(llm_caller=call_gemini)
#     extracted = agent.run(request.file_path)
#     # return extracted
#     return {"extracted_requirements": extracted}
@router.post("/extract")
async def extract_requirements(file: UploadFile = File(...)):
    # Validate file type
    if not file.filename.endswith(('.docx', '.pdf')):
        raise HTTPException(status_code=400, detail="Only .docx and .pdf files are supported")
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
        # Save uploaded content to temp file
        content = await file.read()
        temp_file.write(content)
        temp_path = temp_file.name
    
    try:
        # Process the file
        agent = ExtractorAgent(llm_caller=call_gemini)
        extracted = agent.run(temp_path)
        return {"extracted_requirements": extracted}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.unlink(temp_path)
###########################################################################

class RequirementsRequest(BaseModel):
    extracted_requirements: List[Dict]

@router.post("/classify")
async def classify_requirements(request: RequirementsRequest):
    agent = ClassifierAgent(llm_caller=call_gemini)
    classified = agent.run(request.extracted_requirements)
    return classified

###########################################################################

class ClassifiedRequirementsRequest(BaseModel):
    classified_requirements: List[Dict]

@router.post("/validate")
async def validate_requirements(request: ClassifiedRequirementsRequest):
    agent = ValidationAgent(llm_caller=call_gemini)
    validated = agent.run(request.model_dump())
    return validated

###########################################################################

class ValidatedRequirementsRequest(BaseModel):
    validated_requirements: List[Dict]


@router.post("/user_stories")
async def generate_user_stories(request: ValidatedRequirementsRequest):
    agent = UserStoryAgent(llm_caller=call_gemini)
    user_stories = agent.run(request.model_dump())
    return user_stories

###########################################################################

class FeedbackRequest(BaseModel):
    story: Dict
    feedback: str

@router.post("/update-story")
async def update_story(request: FeedbackRequest):
    agent = UserStoryFeedbackAgent(llm_caller=call_gemini)
    result = agent.run(request.model_dump())
    if not result.get("updated_story"):
        raise HTTPException(status_code=500, detail="Failed to update story from feedback")
    return result

###########################################################################

class UserStoriesRequest(BaseModel):
    user_stories: List[Dict]

@router.post("/gherkin")
async def generate_gherkin(request: UserStoriesRequest):
    agent = GherkinAgent(llm_caller=call_gemini)
    gherkin = agent.run(request.model_dump())
    return gherkin

###########################################################################

@router.post("/test_cases")
async def generate_test_cases(request: UserStoriesRequest):
    agent = TestCaseAgent(llm_caller=call_gemini)
    test_cases = agent.run(request.model_dump())
    print("Generated test cases:", test_cases)
    return test_cases

###########################################################################

class TraceabilityRequest(BaseModel):
    classified_requirements: List[Dict]
    validated_requirements: Optional[List[Dict]]
    user_stories: Optional[List[Dict]]
    gherkin_scenarios: Optional[List[Dict]]
    test_cases: Optional[List[Dict]]

# @router.post("/traceability")
# async def generate_traceability(request: TraceabilityRequest):
#     agent = TraceabilityAgent()  # ✅ no args since class doesn’t support it
#     traceability = agent.run(request.model_dump())
#     return traceability

@router.post("/traceability")
async def generate_traceability(request: TraceabilityRequest):
    inputs = {
        "classified": request.classified_requirements or [],
        "validated": request.validated_requirements or [],
        "user_stories": request.user_stories or [],
        "gherkin_scenarios": request.gherkin_scenarios or [],
        "test_cases": request.test_cases or [],
    }
    agent = TraceabilityAgent()
    traceability = agent.run(inputs)
    return traceability
###########################################################################

class NewTraceabilityRequest(BaseModel):
    user_stories: List[Dict]
    gherkin_scenarios: Optional[List[Dict]] = None
    test_cases: Optional[List[Dict]] = None

@router.post("/newtraceability")
async def generate_new_traceability(request: NewTraceabilityRequest):
    inputs = {
        "user_stories": request.user_stories or [],
        "gherkin_scenarios": request.gherkin_scenarios or [],
        "test_cases": request.test_cases or [],
    }
    agent = NewTraceabilityAgent()
    traceability = agent.run(inputs)
    return traceability

###########################################################################
class UserStoryPayload(BaseModel):
    user_stories: Optional[List[Dict[str, Any]]] = None

@router.post("/export-userstories")
async def export_userstories(payload: UserStoryPayload):
    """
    Accepts JSON payload of user stories and returns an Excel file download.
    """
    result = export_user_stories_to_excel(payload.dict(exclude_none=True))

    if result["status"] == "failed":
        raise HTTPException(status_code=400, detail=result["errors"])

    file_path = result["file_path"]

    if not os.path.exists(file_path):
        raise HTTPException(status_code=500, detail="Excel file not found after export.")

    return FileResponse(
        path=file_path,
        filename=os.path.basename(file_path),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
###########################################################################
class TestCasePayload(BaseModel):
    test_cases: Optional[List[Dict[str, Any]]] = None

@router.post("/export-testcases")
async def export_testcases(payload: TestCasePayload):
    """
    Accepts JSON payload of test cases and returns an Excel file download.
    """
    result = export_test_cases_to_excel(payload.dict(exclude_none=True))

    if result["status"] == "failed":
        raise HTTPException(status_code=400, detail=result["errors"])

    file_path = result["file_path"]

    if not os.path.exists(file_path):
        raise HTTPException(status_code=500, detail="Excel file not found after export.")

    return FileResponse(
        path=file_path,
        filename=os.path.basename(file_path),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
###########################################################################
@router.post("/jira/user_stories")
async def create_user_stories(request: UserStoriesRequest):
    response = post_user_stories_to_jira(request.user_stories)
    if response is None:
        raise HTTPException(status_code=500, detail="Failed to create user stories in Jira")
    return response

###########################################################################

class TestCasesRequest(BaseModel):
    test_cases: List[Dict]

@router.post("/jira/test_cases")
async def create_test_cases(request: TestCasesRequest):
    response = post_test_cases_to_jira(request.test_cases)
    if response is None:
        raise HTTPException(status_code=500, detail="Failed to create test cases in Jira")
    return response
