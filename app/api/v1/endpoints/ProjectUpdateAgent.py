from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import os
import httpx
import logging
import json
from app.api.v1.endpoints.dynamic_agent import generate_text_with_context, _extract_and_parse_json, _repair_project_json

router = APIRouter()
logger = logging.getLogger(__name__)


class ProjectUpdateRequest(BaseModel):
    prompt: str = Field(..., description="User instruction for modifying the project")

    class Config:
        allow_population_by_field_name = True
        extra = "ignore"


async def _fetch_project_by_id(project_id: str) -> dict:
    """Fetch project data from the project service by ID."""
    base_url = os.getenv("PROJECT_SERVICE_URL")
    if not base_url:
        raise HTTPException(status_code=500, detail="PROJECT_SERVICE_URL not configured in environment")
    
    fetch_url = f"{base_url.rstrip('/')}/api/v1/project/get/{project_id}"
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(fetch_url, timeout=10.0)
        
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code, 
                detail=f"Failed to fetch project: {response.text}"
            )
        
        payload = response.json()
        project_data = payload.get("data") if isinstance(payload, dict) else payload
        
        if not project_data:
            raise HTTPException(status_code=404, detail="Project data not found")
        
        return project_data
    
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Error connecting to project service: {str(e)}")


async def _patch_project_to_db(project_id: str, updated_project: dict) -> dict:
    """Send updated project back to the project service."""
    base_url = os.getenv("PROJECT_SERVICE_URL")
    if not base_url:
        raise HTTPException(status_code=500, detail="PROJECT_SERVICE_URL not configured in environment")
    
    patch_url = f"{base_url.rstrip('/')}/api/v1/project/updateFullProjectAnyWhere/{project_id}"
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.patch(
                patch_url, 
                json={"project": updated_project}, 
                timeout=15.0
            )
        
        if response.status_code not in (200, 201, 204):
            raise HTTPException(
                status_code=response.status_code, 
                detail=f"Failed to update project in database: {response.text}"
            )
        
        try:
            return response.json()
        except:
            return {"message": response.text}
    
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Error connecting to project service: {str(e)}")


@router.post("/projectUpdateAgent/{project_id}")
async def projectUpdateAgent(project_id: str, payload: ProjectUpdateRequest):
    """
    Update a project based on user instructions.
    
    This endpoint:
    1. Fetches the current project JSON from the database
    2. Sends it to an LLM with user instructions for modification
    3. Parses the updated JSON returned by the LLM
    4. Patches the updated project back to the database
    
    Args:
        project_id: The unique identifier of the project to update
        payload: Contains the 'prompt' field with user instructions for modifications
    
    Returns:
        {
            "success": true,
            "updated_project": {...},
            "service_response": {...}
        }
    """
    
    # Step 1: Fetch the current project
    logger.info(f"Fetching project: {project_id}")
    project = await _fetch_project_by_id(project_id)
    
    # Step 2: Prepare the LLM prompt
    system_prompt = """You are an expert project manager and JSON transformer. Your task is to modify a project's JSON structure based on user instructions.

Rules:
- Return ONLY valid, complete project JSON
- Preserve all existing fields unless explicitly asked to modify them
- Maintain the same JSON structure and schema
- Do not add explanations, comments, or markdown formatting
- Ensure all required fields are present in the output
- Make only the changes requested in the instruction"""

    user_prompt = f"""Current project JSON:
{json.dumps(project, indent=2, ensure_ascii=False)}

User instruction:
{payload.prompt}

Please modify the project JSON according to the instruction above and return the complete updated JSON."""

    # Step 3: Call the LLM to generate updated project
    logger.info(f"Sending project update request to LLM for project: {project_id}")
    llm_response = await generate_text_with_context(
        context=user_prompt, 
        system_name="ProjectUpdater"
    )
    
    # Step 4: Parse the LLM response
    logger.info("Parsing LLM response")
    updated_project = _extract_and_parse_json(llm_response)
    
    # Step 5: If parsing fails, attempt to repair the JSON
    if not updated_project:
        logger.warning("Initial JSON parsing failed, attempting repair")
        repaired = await _repair_project_json(llm_response, project, strict=False)
        if repaired:
            updated_project = repaired
        else:
            logger.warning("Non-strict repair failed, attempting strict repair")
            repaired_strict = await _repair_project_json(llm_response, project, strict=True)
            if repaired_strict:
                updated_project = repaired_strict
    
    if not updated_project:
        logger.error("Failed to parse JSON from LLM response")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to parse valid JSON from model response. Response (first 500 chars): {llm_response[:500]}"
        )
    
    # Step 6: Patch the updated project back to the database
    logger.info(f"Patching updated project to database: {project_id}")
    service_response = await _patch_project_to_db(project_id, updated_project)
    
    logger.info(f"Project successfully updated: {project_id}")
    
    return {
        "success": True,
        "updated_project": updated_project,
        "service_response": service_response
    }