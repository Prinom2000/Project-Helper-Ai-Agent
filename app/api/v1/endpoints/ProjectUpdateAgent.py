import os
import json
import logging
import asyncio
import re
from typing import Optional
from datetime import datetime, timedelta

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.config import OPENROUTER_API_KEY

router = APIRouter()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class TaskModifyPayload(BaseModel):
    userId: str
    projectId: str
    prompt: str


async def _call_openrouter(messages: list, model: str = "deepseek/deepseek-chat", max_tokens: int = 10000, temperature: float = 0.7) -> str:
    """Call OpenRouter / DeepSeek chat/completions and return the assistant content."""
    key = OPENROUTER_API_KEY
    if not key:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY not configured in environment")

    headers = {
        "Authorization": f"Bearer {key}",
        "HTTP-Referer": "https://yourapp.com",
        "X-Title": "Genie",
    }

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    hosts = [
        "https://openrouter.ai/api/v1/chat/completions",
        "https://api.openrouter.ai/api/v1/chat/completions",
    ]

    last_exc = None
    for host in hosts:
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(host, headers=headers, json=payload, timeout=20.0)
            if resp.status_code != 200:
                logger.warning("OpenRouter call failed: %s %s", resp.status_code, resp.text)
                last_exc = Exception(f"{resp.status_code} {resp.text}")
                continue
            data = resp.json()
            content = None
            if isinstance(data, dict):
                try:
                    content = data["choices"][0]["message"]["content"]
                except Exception:
                    content = data.get("message") or data.get("content")
            if not content:
                logger.warning("OpenRouter response missing content: %s", data)
                last_exc = Exception("No content in response")
                continue
            return str(content).strip()
        except Exception as e:
            logger.exception("OpenRouter request error: %s", e)
            last_exc = e
            continue

    raise HTTPException(status_code=500, detail=f"OpenRouter LLM failure: {str(last_exc)}")


def _extract_and_parse_json(response: str) -> dict | None:
    """Extract and parse JSON from LLM response with robust error handling."""
    if not response:
        return None
    
    response = response.strip()
    
    # Strategy 1: Try direct parse if it starts with {
    if response.startswith("{"):
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass
    
    # Strategy 2: Remove markdown code blocks
    clean = re.sub(r"```(?:json)?\s*\n?", "", response)
    clean = clean.strip()
    
    if clean.startswith("{"):
        try:
            return json.loads(clean)
        except json.JSONDecodeError:
            pass
    
    # Strategy 3: Find JSON object within response
    match = re.search(r"\{[\s\S]*\}", response)
    if match:
        json_str = match.group(0)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    
    # Strategy 4: Try to fix common JSON issues
    try:
        fixed = re.sub(r",(\s*[}\]])", r"\1", response)
        match = re.search(r"\{[\s\S]*\}", fixed)
        if match:
            return json.loads(match.group(0))
    except json.JSONDecodeError:
        pass
    
    # Strategy 5: Find boundaries
    try:
        start = response.find("{")
        end = response.rfind("}")
        if start != -1 and end != -1 and end > start:
            json_str = response[start:end+1]
            return json.loads(json_str)
    except json.JSONDecodeError:
        pass
    
    logger.warning("Could not extract valid JSON from response: %s", response[:200])
    return None


async def fetch_project(base_url: str, project_id: str) -> dict:
    """Fetch project details from the project service."""
    path = f"{base_url.rstrip('/')}/api/v1/project/get/{project_id}"
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(path, timeout=10.0)
        if resp.status_code != 200:
            logger.warning("Failed to fetch project: %s %s", resp.status_code, resp.text)
            raise HTTPException(status_code=404, detail="Project not found")
        data = resp.json()
        # Handle wrapped response
        if isinstance(data, dict) and "data" in data:
            return data["data"]
        return data
    except Exception as e:
        logger.exception("Exception fetching project: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to fetch project: {str(e)}")


async def detect_task_operation_with_llm(user_prompt: str, existing_tasks: list) -> dict:
    """
    Detect if user wants to:
    1. Add a new task
    2. Edit an existing task
    3. Add a subtask to an existing task
    4. Edit a subtask
    
    Returns: {
        "operation": "add_task" | "edit_task" | "add_subtask" | "edit_subtask",
        "target_task": "task name if editing/adding subtask",
        "target_subtask": "subtask title if editing subtask",
        "new_description": "what user wants",
        "reason": "explanation"
    }
    """
    # Build task and subtask list for context
    task_context = []
    for i, task in enumerate(existing_tasks):
        task_name = task.get("task", f"Task {i+1}")
        task_context.append(f"Task: {task_name}")
        subtasks = task.get("subtasks", [])
        for st in subtasks:
            st_title = st.get("title", "")
            if st_title:
                task_context.append(f"  - Subtask: {st_title}")
    
    context_str = "\n".join(task_context)
    
    prompt = f"""Analyze the user's prompt and determine what operation they want to perform.

Existing Tasks and Subtasks:
{context_str}

User Prompt: "{user_prompt}"

Respond ONLY with a JSON object (no other text):
{{
    "operation": "add_task" | "edit_task" | "add_subtask" | "edit_subtask",
    "target_task": "exact task name if editing task or adding/editing subtask, else null",
    "target_subtask": "exact subtask title if editing subtask, else null",
    "new_description": "what the user wants to add/change",
    "reason": "brief explanation"
}}

Operation Guidelines:
- "add_task": User wants to add a NEW task (e.g., "add a task", "create task", "new task")
- "edit_task": User wants to modify an EXISTING task (e.g., "change task X", "update task Y")
- "add_subtask": User wants to add a NEW subtask to an existing task (e.g., "add subtask to X")
- "edit_subtask": User wants to modify an EXISTING subtask (e.g., "change subtask Y in task X")

Target Guidelines:
- For edit_task/add_subtask/edit_subtask: "target_task" must match an existing task name exactly
- For edit_subtask: "target_subtask" must match an existing subtask title exactly
- If targets not found, default to "add_task"
"""

    try:
        messages = [{"role": "user", "content": prompt}]
        result_text = await _call_openrouter(messages, max_tokens=300, temperature=0.5)
        
        json_match = re.search(r"\{[\s\S]*\}", result_text)
        if json_match:
            operation_data = json.loads(json_match.group(0))
            return operation_data
        else:
            logger.warning("Could not extract JSON from operation detection: %s", result_text)
            return {"operation": "add_task", "target_task": None, "target_subtask": None, "new_description": user_prompt, "reason": "Failed to parse LLM response"}
    except Exception as e:
        logger.exception("Error in operation detection: %s", e)
        return {"operation": "add_task", "target_task": None, "target_subtask": None, "new_description": user_prompt, "reason": f"Error: {str(e)}"}


async def generate_task_json_with_llm(operation: str, description: str, existing_task: dict = None) -> dict:
    """
    Generate task or subtask JSON using LLM.
    
    Args:
        operation: "add_task", "edit_task", "add_subtask", "edit_subtask"
        description: What the user wants
        existing_task: The task being edited (for edit operations)
    
    Returns:
        For add_task/edit_task: Complete task object
        For add_subtask/edit_subtask: Single subtask object
    """
    today = datetime.utcnow().date()
    
    if operation == "add_task":
        prompt = f"""Generate a comprehensive task JSON for a new project task.

Description: "{description}"

Return ONLY valid JSON for a single task (no other text, no _id field):
{{
    "task": "task name",
    "details": "task description",
    "taskDueDate": "YYYY-MM-DD (future date)",
    "isDeleted": false,
    "isComplite": false,
    "isArchived": false,
    "isStar": false,
    "subtasks": [
        {{
            "title": "subtask 1",
            "subTaskDueDate": "YYYY-MM-DD",
            "isStar": false,
            "isDeleted": false,
            "isComplite": false
        }},
        {{
            "title": "subtask 2",
            "subTaskDueDate": "YYYY-MM-DD",
            "isStar": false,
            "isDeleted": false,
            "isComplite": false
        }}
    ]
}}

Create 2-3 realistic subtasks. Do NOT include _id fields. Make dates realistic and in the future."""

    elif operation == "edit_task":
        existing_json = json.dumps(existing_task, indent=2) if existing_task else "{}"
        prompt = f"""Generate an updated task JSON based on the user's request.

Current Task:
{existing_json}

User's Change Request: "{description}"

Return ONLY valid JSON for the updated task (no other text, no _id field):
{{
    "task": "updated task name",
    "details": "updated task description",
    "taskDueDate": "YYYY-MM-DD",
    "isDeleted": false,
    "isComplite": false,
    "isArchived": false,
    "isStar": false,
    "subtasks": [
        {{
            "title": "subtask title",
            "subTaskDueDate": "YYYY-MM-DD",
            "isStar": false,
            "isDeleted": false,
            "isComplite": false
        }}
    ]
}}

Do NOT include _id fields in the output. Keep existing subtasks unless user explicitly wants changes."""

    elif operation in ["add_subtask", "edit_subtask"]:
        prompt = f"""Generate a subtask JSON.

Description: "{description}"

Return ONLY valid JSON for a single subtask (no other text, no _id field):
{{
    "title": "subtask title",
    "subTaskDueDate": "YYYY-MM-DD (future date)",
    "isStar": false,
    "isDeleted": false,
    "isComplite": false
}}

Do NOT include _id field. Make date realistic and in the future."""

    else:
        raise HTTPException(status_code=400, detail=f"Unknown operation: {operation}")

    try:
        messages = [{"role": "user", "content": prompt}]
        result_text = await _call_openrouter(messages, max_tokens=500, temperature=0.7)
        
        result_obj = _extract_and_parse_json(result_text)
        if not result_obj:
            logger.error("Failed to generate JSON. Response: %s", result_text[:300])
            raise HTTPException(status_code=500, detail="Failed to generate JSON from LLM")
        
        # Ensure no _id fields in generated content
        if "_id" in result_obj:
            del result_obj["_id"]
        if "subtasks" in result_obj:
            for st in result_obj["subtasks"]:
                if "_id" in st:
                    del st["_id"]
        
        return result_obj
    except Exception as e:
        logger.exception("Error generating JSON: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to generate: {str(e)}")


def merge_task_preserving_ids(existing_task: dict, new_task_data: dict) -> dict:
    """
    Merge new task data into existing task, preserving _id fields.
    
    Strategy:
    1. Preserve task _id
    2. Try to match subtasks by title to preserve their _ids
    3. New subtasks don't get _id
    """
    merged = {**new_task_data}
    
    # Preserve task _id
    if "_id" in existing_task:
        merged["_id"] = existing_task["_id"]
    
    # Handle subtasks
    existing_subtasks = existing_task.get("subtasks", [])
    new_subtasks = new_task_data.get("subtasks", [])
    
    # Create mapping of existing subtasks by title
    existing_by_title = {}
    for st in existing_subtasks:
        title = st.get("title", "").strip().lower()
        if title and "_id" in st:
            existing_by_title[title] = st["_id"]
    
    # Merge subtasks
    merged_subtasks = []
    for new_st in new_subtasks:
        st_copy = {**new_st}
        # Check if this subtask existed before (by matching title)
        title = new_st.get("title", "").strip().lower()
        if title in existing_by_title:
            st_copy["_id"] = existing_by_title[title]
        # Otherwise it's a new subtask - no _id
        merged_subtasks.append(st_copy)
    
    merged["subtasks"] = merged_subtasks
    return merged


@router.post("/modify_task/")
async def modify_task(payload: TaskModifyPayload):
    """
    Add or edit tasks/subtasks in an existing project.
    - Analyzes user prompt to determine operation type
    - Generates appropriate JSON
    - Preserves _id for existing items, omits _id for new items
    - Updates project in database
    """
    base = os.getenv("PROJECT_SERVICE_URL")
    if not base:
        raise HTTPException(status_code=500, detail="PROJECT_SERVICE_URL not configured in environment")

    user_id = payload.userId
    project_id = payload.projectId
    user_prompt = (payload.prompt or "").strip()

    if not user_id or not project_id or not user_prompt:
        raise HTTPException(status_code=400, detail="userId, projectId, and prompt are required")

    # Fetch existing project
    project = await fetch_project(base, project_id)
    
    if project.get("userId") != user_id:
        raise HTTPException(status_code=403, detail="Unauthorized: Project does not belong to user")

    existing_tasks = project.get("tasks", []) or []
    
    # Detect operation type using LLM
    operation_result = await detect_task_operation_with_llm(user_prompt, existing_tasks)
    operation = operation_result.get("operation", "add_task")
    target_task_name = operation_result.get("target_task")
    target_subtask_title = operation_result.get("target_subtask")
    description = operation_result.get("new_description", user_prompt)
    
    logger.info(f"Operation: {operation}, Target Task: {target_task_name}, Target Subtask: {target_subtask_title}")

    updated_tasks = []
    operation_description = ""
    response_summary_data = {}

    if operation == "add_task":
        # Generate new task
        new_task_obj = await generate_task_json_with_llm("add_task", description)
        updated_tasks = existing_tasks + [new_task_obj]
        operation_description = f"Added task: {new_task_obj.get('task')}"
        response_summary_data = new_task_obj

    elif operation == "edit_task":
        # Find and update the task
        found = False
        for task in existing_tasks:
            if task.get("task") == target_task_name:
                # Generate updated task data
                new_task_data = await generate_task_json_with_llm("edit_task", description, task)
                # Merge preserving IDs
                merged_task = merge_task_preserving_ids(task, new_task_data)
                updated_tasks.append(merged_task)
                found = True
                operation_description = f"Updated task: {merged_task.get('task')}"
                response_summary_data = merged_task
            else:
                updated_tasks.append(task)
        
        if not found:
            logger.warning(f"Target task '{target_task_name}' not found. Adding as new task.")
            new_task_obj = await generate_task_json_with_llm("add_task", description)
            updated_tasks = existing_tasks + [new_task_obj]
            operation_description = f"Added task (target not found): {new_task_obj.get('task')}"
            response_summary_data = new_task_obj

    elif operation == "add_subtask":
        # Find task and add subtask
        found = False
        for task in existing_tasks:
            if task.get("task") == target_task_name:
                # Generate new subtask
                new_subtask = await generate_task_json_with_llm("add_subtask", description)
                task_copy = {**task}
                existing_subtasks = task_copy.get("subtasks", [])
                task_copy["subtasks"] = existing_subtasks + [new_subtask]
                updated_tasks.append(task_copy)
                found = True
                operation_description = f"Added subtask '{new_subtask.get('title')}' to task: {task.get('task')}"
                response_summary_data = {"task": task.get('task'), "subtask": new_subtask}
            else:
                updated_tasks.append(task)
        
        if not found:
            logger.warning(f"Target task '{target_task_name}' not found for subtask add.")
            raise HTTPException(status_code=404, detail=f"Task '{target_task_name}' not found")

    elif operation == "edit_subtask":
        # Find task and subtask, then update
        found_task = False
        found_subtask = False
        for task in existing_tasks:
            if task.get("task") == target_task_name:
                found_task = True
                task_copy = {**task}
                updated_subtasks = []
                
                for st in task_copy.get("subtasks", []):
                    if st.get("title") == target_subtask_title:
                        # Generate updated subtask
                        new_st_data = await generate_task_json_with_llm("edit_subtask", description)
                        # Preserve subtask _id
                        if "_id" in st:
                            new_st_data["_id"] = st["_id"]
                        updated_subtasks.append(new_st_data)
                        found_subtask = True
                        operation_description = f"Updated subtask '{new_st_data.get('title')}' in task: {task.get('task')}"
                        response_summary_data = {"task": task.get('task'), "subtask": new_st_data}
                    else:
                        updated_subtasks.append(st)
                
                task_copy["subtasks"] = updated_subtasks
                updated_tasks.append(task_copy)
            else:
                updated_tasks.append(task)
        
        if not found_task:
            raise HTTPException(status_code=404, detail=f"Task '{target_task_name}' not found")
        if not found_subtask:
            raise HTTPException(status_code=404, detail=f"Subtask '{target_subtask_title}' not found in task '{target_task_name}'")

    else:
        raise HTTPException(status_code=400, detail=f"Unknown operation: {operation}")

    # Prepare full project update
    project_update = {
        "goal": project.get("goal"),
        "tasks": updated_tasks,
    }

    # Send update to database
    update_path = f"{base.rstrip('/')}/api/v1/project/updateFullProjectAnyWhere/{project_id}"
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.put(update_path, json=project_update, timeout=10.0)
        
        if resp.status_code not in (200, 201):
            logger.warning("Update project failed: %s %s", resp.status_code, resp.text)
            return {"success": False, "status": resp.status_code, "detail": resp.text}
    except Exception as e:
        logger.exception("Exception updating project: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

    # Build response summary
    response_lines = [operation_description]
    
    if operation in ["add_task", "edit_task"]:
        task_obj = response_summary_data
        if task_obj.get("details"):
            response_lines.append(f"Details: {task_obj.get('details')}")
        if task_obj.get("taskDueDate"):
            response_lines.append(f"Due Date: {task_obj.get('taskDueDate')}")
        
        subtasks = task_obj.get("subtasks", [])
        if subtasks:
            response_lines.append("Subtasks:")
            for st in subtasks:
                st_title = st.get("title", "Subtask")
                st_due = st.get("subTaskDueDate", "")
                if st_due:
                    response_lines.append(f"  - {st_title} (Due: {st_due})")
                else:
                    response_lines.append(f"  - {st_title}")
    
    elif operation in ["add_subtask", "edit_subtask"]:
        st = response_summary_data.get("subtask", {})
        response_lines.append(f"Title: {st.get('title', '')}")
        if st.get("subTaskDueDate"):
            response_lines.append(f"Due Date: {st.get('subTaskDueDate')}")

    response_summary = "\n".join(response_lines)

    return {
        "success": True,
        "operation": operation,
        "data": response_summary_data,
        "summary": response_summary,
        "message": f"Operation completed successfully"
    }