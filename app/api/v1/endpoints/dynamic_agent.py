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
from app.api.v1.endpoints.project_task_question import _fetch_projects_for_user

router = APIRouter()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Global variable to store last message from task_ask route
last_messege = {
    "userId": None,
    "projectId": None,
    "text": None,
}

# Global variable to store last message from subtask_ask route
last_messege_subtask = {
    "userId": None,
    "projectId": None,
    "taskId": None,
    "text": None,
}


class ChatPayload(BaseModel):
    userId: str
    text: str


async def _post_history(base: str, user_id: str, text: str, is_ai: bool, chat_type: str = "ask") -> bool:
    path = f"{base.rstrip('/')}/api/v1/updateHistory/createupdatehistory"
    body = {
        "userId": user_id,
        "text": text,
        "isAi": bool(is_ai),
        "chatType": chat_type,
    }
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(path, json=body, timeout=10.0)
        if resp.status_code in (200, 201):
            return True
        logger.warning("Failed to post history: %s %s", resp.status_code, resp.text)
        return False
    except Exception as e:
        logger.exception("Exception posting history: %s", e)
        return False


async def _fetch_history(base: str, user_id: str, chat_type: str = "ask") -> list:
    path = f"{base.rstrip('/')}/api/v1/updateHistory/find/history/{user_id}/{chat_type}"
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(path, timeout=10.0)
        if resp.status_code != 200:
            logger.warning("Failed to fetch history: %s %s", resp.status_code, resp.text)
            return []
        payload = resp.json()
        # API may wrap under data
        history = payload.get("data") if isinstance(payload, dict) and payload.get("data") else payload
        if isinstance(history, dict):
            return [history]
        if isinstance(history, list):
            return history
        return []
    except Exception as e:
        logger.exception("Exception fetching history: %s", e)
        return []


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
            # Try common response shapes
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


async def generate_text_with_context(context: str, system_name: str = "Genie the project assistant") -> str:
    messages = [
        {"role": "system", "content": f"You are a helpful assistant named {system_name} that helps users with projects."},
        {"role": "user", "content": context},
    ]
    return await _call_openrouter(messages)    


async def detect_project_intent_with_llm(user_text: str, history: str = "") -> dict:
    """
    Use LLM to determine if user's message contains a specific project topic/intent.
    Returns: {"has_specific_topic": bool, "topic": str or None, "num_tasks": int or None, "reason": str}
    """
    prompt = f"""Analyze the user's message and determine if they are requesting a project for a specific topic.

User Message: "{user_text}"

Previous Chat History (if any):
{history if history.strip() else "(No history)"}

Respond ONLY with a JSON object (no other text):
{{
    "has_specific_topic": true/false,
    "topic": "specific topic if identified, or null",
    "num_tasks": number_of_tasks_requested_or_null,
    "reason": "brief explanation"
}}

Guidelines:
- "has_specific_topic": true if the user mentions learning something, building something specific, creating a specific project
  Examples: "I want to learn C++", "now make a project for making a house", "Build an e-commerce site", "Create a mobile app"
- "has_specific_topic": false if the user says something completely generic like "create a project", "make it" WITHOUT specifying what
- If true, extract the exact topic they want (e.g., "Learning C++", "Making a House", "E-commerce Website")
- "num_tasks": If user explicitly mentions task count (e.g., "with 5 tasks", "3 tasks"), extract that number, otherwise null
- If false, return null for both topic and num_tasks"""

    try:
        messages = [{"role": "user", "content": prompt}]
        result_text = await _call_openrouter(messages, max_tokens=10000, temperature=0.5)

        # Extract JSON from response
        json_match = re.search(r"\{[\s\S]*\}", result_text)
        if json_match:
            intent_data = json.loads(json_match.group(0))
            return intent_data
        else:
            logger.warning("Could not extract JSON from intent detection: %s", result_text)
            return {"has_specific_topic": False, "topic": None, "num_tasks": None, "reason": "Failed to parse LLM response"}
    except Exception as e:
        logger.exception("Error in LLM intent detection: %s", e)
        return {"has_specific_topic": False, "topic": None, "num_tasks": None, "reason": f"Error: {str(e)}"}


def _extract_and_parse_json(response: str) -> dict | None:
    """
    Extract and parse JSON from LLM response with robust error handling.
    Tries multiple strategies to extract valid JSON.
    """
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
        # Remove trailing commas before closing braces/brackets
        fixed = re.sub(r",(\s*[}\]])", r"\1", response)
        match = re.search(r"\{[\s\S]*\}", fixed)
        if match:
            return json.loads(match.group(0))
    except json.JSONDecodeError:
        pass
    
    # Strategy 5: Find the last closing brace and try to parse from first opening brace
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


async def _repair_project_json(raw_response: str, sample: dict, strict: bool = False) -> dict | None:
    """Ask the LLM to reformat or extract valid JSON from a raw LLM response.
    Returns the parsed dict or None on failure."""
    if not raw_response:
        return None

    sys_msg = "You are a JSON formatter. Extract and return ONLY the JSON object that matches the example schema provided. Do NOT return any explanatory text."
    user_msg = (
        "The following text may contain a JSON object, surrounding text, or markdown code fences. "
        "Please extract and return ONLY the valid JSON object that matches this example exactly.\n\n"
        f"Example structure:\n{json.dumps(sample, ensure_ascii=False, indent=2)}\n\n"
        "Here is the original response to fix:\n" + raw_response
    )

    # If strict, emphasize no extra text and lower temperature
    temp = 0.0 if strict else 0.2

    try:
        messages = [{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}]
        repaired = await _call_openrouter(messages, max_tokens=10000, temperature=temp)
        parsed = _extract_and_parse_json(repaired)
        if parsed:
            return parsed
        # If the repaired text didn't parse, try heuristics on the repaired text
        parsed2 = _extract_and_parse_json(repaired + "\n")
        if parsed2:
            return parsed2
    except Exception as e:
        logger.warning("JSON repair attempt failed: %s", e)
    return None


def _normalize_project_obj(project: dict, min_tasks: int = 3, max_tasks: int = 10) -> dict:
    """Ensure the project matches the required schema and fill reasonable dates if missing.
    Enforces a maximum of `max_tasks` tasks and will trim extras if present."""
    today = datetime.utcnow().date()

    # Ensure goal
    goal = project.get("goal") or project.get("project_goal") or "(No goal provided)"

    # Clamp min_tasks to a sensible range and respect the maximum
    min_tasks = max(0, min(min_tasks, max_tasks))

    tasks = project.get("tasks") or []
    if not isinstance(tasks, list):
        tasks = [tasks]

    # Ensure minimum number of tasks (no random), but never exceed max_tasks
    if len(tasks) < min_tasks:
        for i in range(len(tasks), min_tasks):
            tasks.append({"task": f"Task {i + 1} for {goal}"})

    # Trim to maximum number of tasks if LLM produced too many
    if len(tasks) > max_tasks:
        logger.warning("Trimming tasks from %d to max %d", len(tasks), max_tasks)
        tasks = tasks[:max_tasks]

    normalized_tasks = []
    for i, t in enumerate(tasks):
        if not isinstance(t, dict):
            t = {"task": str(t)}
        task_name = t.get("task") or t.get("title") or f"Task {i+1}"
        details = t.get("details") or t.get("description") or ""
        # task due date: ensure a valid future date
        task_due = t.get("taskDueDate") or t.get("due")

        def _ensure_future_date(d_str, default_days):
            if not d_str:
                return (today + timedelta(days=default_days)).strftime("%Y-%m-%d")
            try:
                d = datetime.strptime(str(d_str), "%Y-%m-%d").date()
                if d <= today:
                    return (today + timedelta(days=default_days)).strftime("%Y-%m-%d")
                return d.strftime("%Y-%m-%d")
            except Exception:
                return (today + timedelta(days=default_days)).strftime("%Y-%m-%d")

        task_due_date = _ensure_future_date(task_due, (i + 1) * 14)
        is_star = bool(t.get("isStar")) if t.get("isStar") is not None else False

        subtasks = t.get("subtasks") or []
        if not isinstance(subtasks, list):
            subtasks = [subtasks]
        # ensure at least 1-3 subtasks
        if len(subtasks) == 0:
            subtasks = [
                {"title": f"{task_name} - subtask 1"},
                {"title": f"{task_name} - subtask 2"},
            ]

        normalized_subtasks = []
        try:
            task_due_date_dt = datetime.strptime(task_due_date, "%Y-%m-%d").date()
        except Exception:
            task_due_date_dt = today + timedelta(days=(i + 1) * 14)

        for j, st in enumerate(subtasks):
            if not isinstance(st, dict):
                st = {"title": str(st)}
            st_title = st.get("title") or st.get("task") or f"Subtask {j+1}"
            st_due_raw = st.get("subTaskDueDate") or st.get("subTask_due")

            # compute candidate due as earlier than task but still in the future
            candidate = task_due_date_dt - timedelta(days=(j + 1) * 3)

            if st_due_raw:
                try:
                    st_d = datetime.strptime(str(st_due_raw), "%Y-%m-%d").date()
                    if st_d <= today:
                        # ensure it's future and not after task due
                        if candidate > today:
                            st_due_date = candidate.strftime("%Y-%m-%d")
                        else:
                            st_due_date = (today + timedelta(days=(i + 1) * 7 + (j + 1) * 3)).strftime("%Y-%m-%d")
                    else:
                        st_due_date = st_d.strftime("%Y-%m-%d")
                except Exception:
                    if candidate > today:
                        st_due_date = candidate.strftime("%Y-%m-%d")
                    else:
                        st_due_date = (today + timedelta(days=(i + 1) * 7 + (j + 1) * 3)).strftime("%Y-%m-%d")
            else:
                if candidate > today:
                    st_due_date = candidate.strftime("%Y-%m-%d")
                else:
                    st_due_date = (today + timedelta(days=(i + 1) * 7 + (j + 1) * 3)).strftime("%Y-%m-%d")

            st_star = bool(st.get("isStar")) if st.get("isStar") is not None else False
            normalized_subtasks.append({"title": st_title, "subTaskDueDate": st_due_date, "isStar": st_star})

        normalized_tasks.append({
            "task": task_name,
            "details": details,
            "taskDueDate": task_due_date,
            "isStar": is_star,
            "subtasks": normalized_subtasks,
        })

    normalized = {"goal": goal, "tasks": normalized_tasks}
    return normalized


@router.post("/ask_helper/")
async def ask_helper(payload: ChatPayload):
    """Chat assistant (Genie) for user projects. Records both user and AI messages to updateHistory (chatType='ask')."""
    base = os.getenv("PROJECT_SERVICE_URL")
    if not base:
        raise HTTPException(status_code=500, detail="PROJECT_SERVICE_URL not configured in environment")

    user_id = payload.userId
    user_text = (payload.text or "").strip()
    if not user_id or not user_text:
        raise HTTPException(status_code=400, detail="userId and text are required")

    # Save user message in history
    await _post_history(base, user_id, user_text, is_ai=False, chat_type="ask")

    # Fetch user's projects to create context
    projects = await _fetch_projects_for_user(user_id)
    project_summary = json.dumps(projects, ensure_ascii=False, indent=2)

    context = (
        "You are Genie, a friendly project helper that understands the user's projects. "
        "Use the project information below to assist the user and give practical advice, suggestions and tasks."
        "\n\nPROJECTS:\n" + project_summary + f"\n\nUser: {user_text}\n\nRespond concisely as Genie."
    )

    # Call LLM
    reply = await generate_text_with_context(context, system_name="Genie")

    # Save AI response to history
    await _post_history(base, user_id, reply, is_ai=True, chat_type="ask")

    return {"response": reply}


@router.post("/create_project/")
async def create_project(payload: ChatPayload):
    """
    Create a full project JSON from ask history or explicit user request.
    - If user specifies a topic (e.g., "I want to learn C++"), create project for that topic.
    - If user says something generic (e.g., "now make the project"), use ask_helper history.
    """
    base = os.getenv("PROJECT_SERVICE_URL")
    if not base:
        raise HTTPException(status_code=500, detail="PROJECT_SERVICE_URL not configured in environment")

    user_id = payload.userId
    user_text = (payload.text or "").strip()
    if not user_id or not user_text:
        raise HTTPException(status_code=400, detail="userId and text are required")

    # Save the user's create request to history
    await _post_history(base, user_id, user_text, is_ai=False, chat_type="create")

    # Fetch ask history for context - extract latest message directly
    history = await _fetch_history(base, user_id, chat_type="ask")
    print(f"Fetched ask history: {history}")
    latest_message = ""
    if history and isinstance(history, list):
        # Get the last item from history for context
        latest_item = history[-1]
        if isinstance(latest_item, dict):
            latest_message = latest_item.get("text") or latest_item.get("message") or ""
            print(f"Latest history item not a dict: {latest_item}")
        else:
            latest_message = str(latest_item)
            print(f"Latest history item not a dict: {latest_item}")
        logger.info(f"Retrieved latest message from ask history: {latest_message}")
    
    # For intent detection, use both current user text and latest context
    aggregated = latest_message if latest_message else user_text

    # Use LLM to detect if user has a specific topic/intent
    intent_result = await detect_project_intent_with_llm(user_text, aggregated)
    has_specific_topic = intent_result.get("has_specific_topic", False)
    topic = intent_result.get("topic")
    num_tasks = intent_result.get("num_tasks")
    
    logger.info(f"Intent detection for user {user_id}: has_specific_topic={has_specific_topic}, topic={topic}, num_tasks={num_tasks}")

    # Determine which context to use
    if has_specific_topic and topic:
        # User specified a specific project topic — ignore history
        used_num_tasks = None
        try:
            used_num_tasks = int(num_tasks) if num_tasks is not None else None
        except Exception:
            used_num_tasks = None

        if used_num_tasks and used_num_tasks > 0:
            if used_num_tasks > 10:
                logger.info("Requested %s tasks exceeds max 10; capping at 10", used_num_tasks)
                used_num_tasks = 10
                task_count_text = f" with {num_tasks} tasks (capped to 10)"
            else:
                task_count_text = f" with {used_num_tasks} tasks"
        else:
            task_count_text = ""

        prompt_intro = f"User wants to create a project for: '{topic}'{task_count_text}. Create a comprehensive project JSON based on this specific topic."
        
        if used_num_tasks and used_num_tasks > 0:
            prompt_context = f"Project Topic: {topic}\nNumber of Tasks: {used_num_tasks}\n\nGenerate a realistic, well-structured project with exactly {used_num_tasks} tasks and relevant subtasks for each task. Ensure dates use YYYY-MM-DD."
        else:
            prompt_context = f"Project Topic: {topic}\n\nGenerate a realistic, well-structured project with at least 3 tasks and relevant subtasks. Ensure dates use YYYY-MM-DD."
    else:
        # No specific topic — use ask history
        if not aggregated.strip():
            return {"created": False, "reason": "No ask history available and no specific project topic mentioned. Please discuss your project idea in the chat first, or specify what you want to create."}
        
        prompt_intro = "Based on the following conversation, produce a JSON object representing a full project. Use the conversation to infer the user's goal and tasks."
        prompt_context = "Conversation and project info:\n" + aggregated

    # Provide an explicit sample structure
    sample = {
  "userId": "605c72ef1532076f72d9a132",
  "goal": "Launch New Website",
  "tasks": [
    {
      "title": "Design Homepage",
      "description": "Design the layout and structure for the homepage",
      "compliteTarget": "2026-03-02",
      "subtasks": [
        {
          "title": "Wireframe Homepage",
          "description": "Create wireframe of homepage layout",
          "compliteTarget": "2026-02-10"
        },
        {
          "title": "UI Design",
          "description": "Design visual elements like buttons, typography, and colors",
          "compliteTarget": "2026-02-20"
        }
      ]
    },
    {
      "title": "Develop Backend",
      "description": "Develop backend services to support website functionality",
      "compliteTarget": "2026-03-12",
      "subtasks": [
        {
          "title": "Set up Database",
          "description": "Set up and configure the database for storing website data",
          "compliteTarget": "2026-02-15"
        },
        {
          "title": "API Development",
          "description": "Develop REST APIs for user interaction and data management",
          "compliteTarget": "2026-02-25"
        }
      ]
    }
  ]
 }


    prompt = (
        prompt_intro
        + "\nReturn ONLY valid JSON that exactly matches the structure and keys of the example below (no extra text, no markdown, no explanation):\n\n"
        + json.dumps(sample, ensure_ascii=False, indent=2)
        + "\n\n"
        + prompt_context
        + "\n\nGenerate a realistic project with relevant subtasks. Ensure dates use YYYY-MM-DD."
    )

    gen = await generate_text_with_context(prompt, system_name="Genie")
    
    logger.info(f"LLM response for project creation: {gen}")

    # Extract and parse JSON using robust function
    project_obj = _extract_and_parse_json(gen)

    # If initial parsing failed, attempt to repair via the LLM (up to 2 attempts)
    if not project_obj:
        logger.warning("Initial parse failed, attempting to repair JSON via LLM...")
        repaired = await _repair_project_json(gen, sample, strict=False)
        if repaired:
            project_obj = repaired
            logger.info("Successfully repaired JSON on attempt 1")
        else:
            logger.warning("Repair attempt 1 failed, attempting strict repair...")
            repaired2 = await _repair_project_json(gen, sample, strict=True)
            if repaired2:
                project_obj = repaired2
                logger.info("Successfully repaired JSON on attempt 2")

    if not project_obj:
        logger.error("Failed to extract JSON after repair attempts. Full LLM response: %s", gen)
        # Provide truncated raw response to aid debugging
        snippet = gen[:1000] if gen else ""
        raise HTTPException(status_code=500, detail=f"Failed to parse project JSON from LLM response. Raw response (truncated): {snippet}")

    # Normalize and enforce schema with specific task count if provided
    # Determine minimum task count to enforce, using capped requested value if present
    if 'used_num_tasks' in locals() and used_num_tasks and used_num_tasks > 0:
        min_tasks_count = min(used_num_tasks, 10)
    else:
        min_tasks_count = 3

    project_obj = _normalize_project_obj(project_obj, min_tasks=min_tasks_count)

    # Transform normalized project to requested schema and include userId
    create_payload = {
        "userId": user_id,
        "goal": project_obj.get("goal"),
        "tasks": []
    }

    for t in project_obj.get("tasks", []) or []:
        title = t.get("task") or t.get("title") or "Untitled Task"
        description = t.get("details") or t.get("description") or ""
        complite_target = t.get("taskDueDate") or t.get("compliteTarget") or None
        subtasks_out = []
        for st in t.get("subtasks", []) or []:
            st_title = st.get("title") or st.get("task") or "Subtask"
            st_description = st.get("description") or ""
            st_complite = st.get("subTaskDueDate") or st.get("compliteTarget") or None
            subtasks_out.append({
                "title": st_title,
                "description": st_description,
                "compliteTarget": st_complite,
            })

        create_payload["tasks"].append({
            "title": title,
            "description": description,
            "compliteTarget": complite_target,
            "subtasks": subtasks_out,
        })

    # Send to updateProject/createProject endpoint
    create_path = f"{base.rstrip('/')}/api/v1/updateProject/createProject"
    logger.info(f"Sending full project payload to DB: {json.dumps(create_payload, ensure_ascii=False, indent=2)}")
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(create_path, json=create_payload, timeout=10.0)
        if resp.status_code not in (200, 201):
            logger.warning("CreateProject failed: %s %s", resp.status_code, resp.text)
            return {"created": False, "status": resp.status_code, "detail": resp.text}
    except Exception as e:
        logger.exception("Exception creating project: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

    # Build a concise dynamic summary
    goal = project_obj.get("goal") or "(No goal provided)"
    lines = [f"Project created — Goal: {goal}"]
    tasks = project_obj.get("tasks", []) or []
    if tasks:
        lines.append("Tasks:")
        for t in tasks:
            if isinstance(t, dict):
                task_name = t.get("task") or t.get("title") or "Untitled Task"
                details = t.get("details")
                due = t.get("taskDueDate") or t.get("due")
                is_star = t.get("isStar")
                line = f"- {task_name}"
                extras = []
                if details:
                    extras.append(str(details))
                if due:
                    extras.append(f"Due: {due}")
                if is_star:
                    extras.append("★")
                if extras:
                    line += " (" + "; ".join(extras) + ")"
                lines.append(line)
                for st in t.get("subtasks", []) or []:
                    st_title = st.get("title") or st.get("task") or "Subtask"
                    st_due = st.get("subTaskDueDate") or st.get("subTask_due")
                    st_star = st.get("isStar")
                    st_line = f"  - {st_title}"
                    st_extras = []
                    if st_due:
                        st_extras.append(f"Due: {st_due}")
                    if st_star:
                        st_extras.append("★")
                    if st_extras:
                        st_line += " (" + "; ".join(st_extras) + ")"
                    lines.append(st_line)
            else:
                lines.append(f"- {t}")

    summary = "\n".join(lines) + "\n\nFeel free to check out the project list. If you can't see it yet, please delete the entire chat and provide me the prompt again."

    # Save dynamic AI confirmation to history
    await _post_history(base, user_id, summary, is_ai=True, chat_type="create")

    return {"response": summary, "created": True, "project": create_payload}


# ========================
# NEW PYDANTIC MODELS
# ========================

class TaskPayload(BaseModel):
    userId: str
    projectId: str
    text: str


class TaskCreatePayload(BaseModel):
    userId: str
    projectId: str
    query: str


class SubtaskPayload(BaseModel):
    userId: str
    projectId: str
    taskId: str
    text: str


# ========================
# HELPER FUNCTIONS
# ========================

async def _fetch_project_data(base: str, project_id: str) -> dict | None:
    """Fetch project data from DB: {baseUrl}/api/v1/updateProject/getProject/:projectId"""
    path = f"{base.rstrip('/')}/api/v1/updateProject/getProject/{project_id}"
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(path, timeout=10.0)
        if resp.status_code != 200:
            logger.warning("Failed to fetch project data: %s %s", resp.status_code, resp.text)
            return None
        return resp.json()
    except Exception as e:
        logger.exception("Exception fetching project data: %s", e)
        return None


async def _fetch_subtask_data(base: str, task_id: str, project_id: str) -> dict | None:
    """Fetch subtask data from DB: {baseUrl}/api/v1/updateProject/tasks/:taskId/parents/:projectId"""
    path = f"{base.rstrip('/')}/api/v1/updateProject/tasks/{task_id}/parents/{project_id}"
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(path, timeout=10.0)
        if resp.status_code != 200:
            logger.warning("Failed to fetch subtask data: %s %s", resp.status_code, resp.text)
            return None
        return resp.json()
    except Exception as e:
        logger.exception("Exception fetching subtask data: %s", e)
        return None


async def _fetch_project_chat_history(base: str, user_id: str, project_or_task_id: str, chat_type: str = "ask") -> list:
    """Fetch chat history from DB: {baseUrl}/api/v1/projectChatHistory/getProjectChat/:userId/:projectOrTaskId/:chatType"""
    path = f"{base.rstrip('/')}/api/v1/projectChatHistory/getProjectChat/{user_id}/{project_or_task_id}/{chat_type}"
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(path, timeout=10.0)
        if resp.status_code != 200:
            logger.warning("Failed to fetch project chat history: %s %s", resp.status_code, resp.text)
            return []
        payload = resp.json()
        # API may wrap under data
        history = payload.get("data") if isinstance(payload, dict) and payload.get("data") else payload
        if isinstance(history, dict):
            return [history]
        if isinstance(history, list):
            return history
        return []
    except Exception as e:
        logger.exception("Exception fetching project chat history: %s", e)
        return []


async def _save_to_project_chat_history(base: str, user_id: str, project_or_task_id: str, text: str, is_ai: bool, chat_type: str = "ask") -> bool:
    """Save message to project chat history: {baseUrl}/api/v1/projectChatHistory/create"""
    path = f"{base.rstrip('/')}/api/v1/projectChatHistory/create"
    body = {
        "userId": user_id,
        "projectOrTaskId": project_or_task_id,
        "isAi": bool(is_ai),
        "chatType": chat_type,
        "text": text,
    }
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(path, json=body, timeout=10.0)
        if resp.status_code in (200, 201):
            return True
        logger.warning("Failed to save to project chat history: %s %s", resp.status_code, resp.text)
        return False
    except Exception as e:
        logger.exception("Exception saving to project chat history: %s", e)
        return False


async def _detect_creation_intent_with_llm(user_text: str, task_context: str = "") -> dict:
    """Use LLM to detect if user query is a command for making task, subtask, or general conversation.
    Returns: {"intent": "task" | "subtask" | "general", "reason": str}"""
    prompt = f"""Analyze the user's message and determine their intent.

Task Context:
{task_context if task_context.strip() else "(No specific task context)"}

User Message: "{user_text}"

Respond ONLY with a JSON object (no other text):
{{
    "intent": "task" | "subtask" | "general",
    "reason": "brief explanation"
}}

Guidelines:
- "task": User is explicitly asking to create/make/add a new TASK (e.g., "make a new task", "add task", "create task", "I need a new task for this project")
- "subtask": User is explicitly asking to create/make/add a SUBTASK under current task (e.g., "make subtask", "add subtask", "create subtask under this", "break this into subtasks")
- "general": User is asking a general question, seeking help, or having a conversation about the task (e.g., "what should I do?", "how do I proceed?", "tell me about this", "help me understand")

Be precise: Only use "task" or "subtask" if it's a clear CREATE/MAKE/ADD command. Otherwise use "general"."""

    try:
        messages = [{"role": "user", "content": prompt}]
        result_text = await _call_openrouter(messages, max_tokens=10000, temperature=0.5)

        # Extract JSON from response
        json_match = re.search(r"\{[\s\S]*\}", result_text)
        if json_match:
            intent_data = json.loads(json_match.group(0))
            return intent_data
        else:
            logger.warning("Could not extract JSON from intent detection: %s", result_text)
            return {"intent": "general", "reason": "Failed to parse LLM response"}
    except Exception as e:
        logger.exception("Error in LLM intent detection: %s", e)
        return {"intent": "general", "reason": f"Error: {str(e)}"}


async def _detect_subtask_intent_with_deepseek(user_query: str, subtask_context: str = "", task_context: str = "") -> dict:
    """
    Use DeepSeek LLM to detect if user's message is a command for making a subtask or general chat.
    Returns: {"intent": "create_subtask" | "general_chat", "reason": str}
    """
    prompt = f"""Analyze the user's message and determine their intent.

Last subtask conversation message:
{subtask_context.strip() if subtask_context and subtask_context.strip() else "(No previous subtask message)"}

Previous task conversation (as fallback):
{task_context.strip() if task_context and task_context.strip() else "(No task message)"}

Current User Query: "{user_query}"

Respond ONLY with a JSON object (no other text):
{{
    "intent": "create_subtask" | "general_chat",
    "reason": "brief explanation"
}}

Guidelines:
- "create_subtask": User is explicitly asking to create/make/add a new SUBTASK (e.g., "make a subtask", "add subtask", "create subtask", "I need a subtask", "break this down")
- "general_chat": User is asking a general question, seeking help, or having a conversation (e.g., "what should I do?", "how do I proceed?", "tell me about this", "help me understand")

Be precise: Only use "create_subtask" if it's a clear CREATE/MAKE/ADD command. Otherwise use "general_chat"."""

    try:
        messages = [{"role": "user", "content": prompt}]
        result_text = await _call_openrouter(messages, model="deepseek/deepseek-chat", max_tokens=1000, temperature=0.3)

        # Extract JSON from response
        json_match = re.search(r"\{[\s\S]*\}", result_text)
        if json_match:
            intent_data = json.loads(json_match.group(0))
            return intent_data
        else:
            logger.warning("Could not extract JSON from subtask intent detection: %s", result_text)
            return {"intent": "general_chat", "reason": "Failed to parse LLM response"}
    except Exception as e:
        logger.exception("Error in subtask intent detection: %s", e)
        return {"intent": "general_chat", "reason": f"Error: {str(e)}"}


async def _generate_subtask_from_query(user_query: str, parent_task_data: dict, parent_task_id: str, project_id: str, user_id: str, subtask_context: str = "") -> dict | None:
    """
    Use DeepSeek LLM to generate a subtask object from user query.
    Returns: {"parentTaskId": str, "title": str, "userId": str, "projectId": str, "description": str, "compliteTarget": str} or None on failure
    """
    prompt = f"""Based on the parent task information and user query, generate a subtask object.

Parent Task Data:
{json.dumps(parent_task_data, ensure_ascii=False, indent=2)}

Previous subtask conversation:
{subtask_context.strip() if subtask_context and subtask_context.strip() else "(No previous subtask context)"}

Current User Query: "{user_query}"

Create a subtask with the following JSON structure. Use the query and task context to fill in meaningful values:
{{
    "parentTaskId": "{parent_task_id}",
    "title": "Short subtask title (max 100 chars)",
    "userId": "{user_id}",
    "projectId": "{project_id}",
    "description": "Detailed description of what needs to be done",
    "compliteTarget": "YYYY-MM-DD format date for completion deadline"
}}

IMPORTANT RULES:
- MUST include all 6 fields exactly as shown above in this exact order
- MUST use EXACT parentTaskId='{parent_task_id}', userId='{user_id}', projectId='{project_id}' (do NOT change them)
- title: Concise, action-oriented summary of the subtask based on user query
- description: Detailed explanation of what the subtask involves
- compliteTarget: A reasonable completion date in YYYY-MM-DD format (should be before or same as parent task date)

Return ONLY the JSON object with all 6 fields, no other text."""

    try:
        messages = [{"role": "user", "content": prompt}]
        result_text = await _call_openrouter(messages, model="deepseek/deepseek-chat", max_tokens=1000, temperature=0.5)
        
        logger.info(f"LLM response for subtask generation: {result_text}")

        # Extract JSON from response
        json_match = re.search(r"\{[\s\S]*\}", result_text)
        if json_match:
            subtask_data = json.loads(json_match.group(0))
            
            # Ensure all required fields are present with correct values
            subtask_data["parentTaskId"] = parent_task_id
            subtask_data["userId"] = user_id
            subtask_data["projectId"] = project_id
            
            # Ensure title exists
            if "title" not in subtask_data or not subtask_data["title"]:
                subtask_data["title"] = user_query[:100]
            
            # Ensure description exists
            if "description" not in subtask_data:
                subtask_data["description"] = ""
            
            # Ensure compliteTarget exists and is in YYYY-MM-DD format
            if "compliteTarget" not in subtask_data or not subtask_data["compliteTarget"]:
                # Default to tomorrow
                tomorrow = (datetime.utcnow().date() + timedelta(days=1)).strftime("%Y-%m-%d")
                subtask_data["compliteTarget"] = tomorrow
            
            logger.info(f"Generated subtask object: {json.dumps(subtask_data, ensure_ascii=False, indent=2)}")
            return subtask_data
        else:
            logger.warning("Could not extract JSON from subtask generation: %s", result_text)
            return None
    except Exception as e:
        logger.exception("Error in subtask generation: %s", e)
        return None


async def _push_subtask_to_database(base: str, subtask_data: dict) -> dict | None:
    """
    Push subtask to database: POST {baseUrl}/api/v1/updateProject/createSubtaskUnerTaskOrSubtask
    """
    path = f"{base.rstrip('/')}/api/v1/updateProject/createSubtaskUnerTaskOrSubtask"
    
    logger.info(f"Pushing subtask to database at: {path}")
    logger.info(f"Subtask payload: {json.dumps(subtask_data, ensure_ascii=False, indent=2)}")
    
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(path, json=subtask_data, timeout=10.0)
        
        logger.info(f"Database response status: {resp.status_code}")
        logger.info(f"Database response body: {resp.text}")
        
        if resp.status_code in (200, 201):
            return resp.json()
        
        logger.warning("Failed to push subtask to database: status=%s, response=%s", resp.status_code, resp.text)
        return None
    except Exception as e:
        logger.exception("Exception pushing subtask to database: %s", e)
        return None


async def _generate_subtask_feedback_response(user_query: str, parent_task_data: dict, subtask_context: str = "") -> str:
    """
    Generate a feedback/response message for user query without creating a subtask.
    Uses LLM to provide helpful response based on task context.
    Returns: feedback message string
    """
    prompt = f"""Provide helpful feedback and response to the user's query based on the task context.

Parent Task Data:
{json.dumps(parent_task_data, ensure_ascii=False, indent=2)}

Previous subtask conversation:
{subtask_context.strip() if subtask_context and subtask_context.strip() else "(No previous conversation)"}

Current User Query: "{user_query}"

Provide a helpful, concise response to the user's question or message. Be practical and reference the task context where relevant.
Do not create a subtask, just provide helpful feedback and guidance based on their question."""

    try:
        messages = [{"role": "user", "content": prompt}]
        feedback = await _call_openrouter(messages, model="deepseek/deepseek-chat", max_tokens=1500, temperature=0.7)
        return feedback.strip()
    except Exception as e:
        logger.exception("Error in subtask feedback generation: %s", e)
        return f"I encountered an error while processing your query: {str(e)}"


async def _generate_subtask_creation_response(user_query: str, subtask_data: dict, parent_task_data: dict) -> str:
    """
    Generate a friendly response message confirming subtask creation.
    Uses LLM to provide a helpful confirmation message.
    Returns: response message string
    """
    prompt = f"""The user has just created a new subtask. Generate a friendly, professional response message confirming the subtask creation.

User Query: "{user_query}"

Created Subtask:
{json.dumps(subtask_data, ensure_ascii=False, indent=2)}

Parent Task Context:
{json.dumps(parent_task_data, ensure_ascii=False, indent=2)}

Generate a concise, friendly response message that:
1. Confirms the subtask has been created successfully
2. Highlights key details about the subtask (title and due date)
3. Offers brief encouragement or next steps
Keep it to 2-3 sentences."""

    try:
        messages = [{"role": "user", "content": prompt}]
        response = await _call_openrouter(messages, model="deepseek/deepseek-chat", max_tokens=500, temperature=0.7)
        return response.strip()
    except Exception as e:
        logger.exception("Error in subtask creation response generation: %s", e)
        subtask_title = subtask_data.get("title", "Subtask")
        return f"Subtask '{subtask_title}' has been created successfully!"


# ========================
# ROUTES
# ========================

@router.post("/task_ask/")
async def task_ask_route(payload: TaskPayload):
    """
    Ask question about a project.
    - Fetch project data from DB
    - Save user message to chat history
    - Use LLM to answer based on project context
    - Save AI response to chat history
    - Store last message in global variable
    """
    global last_messege
    
    base = os.getenv("PROJECT_SERVICE_URL")
    if not base:
        raise HTTPException(status_code=500, detail="PROJECT_SERVICE_URL not configured in environment")

    user_id = payload.userId
    project_id = payload.projectId
    user_text = (payload.text or "").strip()

    if not user_id or not project_id or not user_text:
        raise HTTPException(status_code=400, detail="userId, projectId, and text are required")

    # Store last message in global variable
    last_messege = {
        "userId": user_id,
        "projectId": project_id,
        "text": user_text,
    }

    # Save user message to chat history
    await _save_to_project_chat_history(base, user_id, project_id, user_text, is_ai=False, chat_type="ask")

    # Fetch project data from DB
    project_data = await _fetch_project_data(base, project_id)
    if not project_data:
        raise HTTPException(status_code=500, detail="Failed to fetch project data from database")

    # Build context with project data
    project_json = json.dumps(project_data, ensure_ascii=False, indent=2)
    context = (
        "You are a helpful project assistant. Answer the user's question about the following project. "
        "Provide concise and practical answers based on the project information.\n\n"
        f"PROJECT DATA:\n{project_json}\n\n"
        f"User Question: {user_text}\n\n"
        "Provide a helpful response."
    )

    # Call LLM to generate response
    reply = await generate_text_with_context(context, system_name="Project Assistant")

    # Save AI response to chat history
    await _save_to_project_chat_history(base, user_id, project_id, reply, is_ai=True, chat_type="ask")

    return {"user_text": user_text, "response": reply}

async def _detect_task_intent_with_deepseek(user_query: str, last_message: str = "") -> dict:
    """
    Use DeepSeek LLM to detect if user's message is a command for making a task or general chat.
    Returns: {"intent": "create_task" | "general_chat", "reason": str}
    """
    prompt = f"""Analyze the user's message and determine their intent.

Last conversation message:
{last_message.strip() if last_message and last_message.strip() else "(No previous message)"}

Current User Query: "{user_query}"

Respond ONLY with a JSON object (no other text):
{{
    "intent": "create_task" | "general_chat",
    "reason": "brief explanation"
}}

Guidelines:
- "create_task": User is explicitly asking to create/make/add a new TASK (e.g., "make a new task", "add task", "create task", "I need a new task", "make a task for this")
- "general_chat": User is asking a general question, seeking help, or having a conversation (e.g., "what should I do?", "how do I proceed?", "tell me about this")

Be precise: Only use "create_task" if it's a clear CREATE/MAKE/ADD command. Otherwise use "general_chat"."""

    try:
        messages = [{"role": "user", "content": prompt}]
        result_text = await _call_openrouter(messages, model="deepseek/deepseek-chat", max_tokens=1000, temperature=0.3)

        # Extract JSON from response
        json_match = re.search(r"\{[\s\S]*\}", result_text)
        if json_match:
            intent_data = json.loads(json_match.group(0))
            return intent_data
        else:
            logger.warning("Could not extract JSON from task intent detection: %s", result_text)
            return {"intent": "general_chat", "reason": "Failed to parse LLM response"}
    except Exception as e:
        logger.exception("Error in task intent detection: %s", e)
        return {"intent": "general_chat", "reason": f"Error: {str(e)}"}


async def _generate_task_from_query(user_query: str, project_data: dict, last_message: str = "") -> dict | None:
    """
    Use DeepSeek LLM to generate a task object from user query.
    Returns: {"title": str, "description": str, "compliteTarget": str} or None on failure
    """
    prompt = f"""Based on the project information and user query, generate a task object.

Project Data:
{json.dumps(project_data, ensure_ascii=False, indent=2)}

Last conversation message:
{last_message.strip() if last_message and last_message.strip() else "(No previous message)"}

Current User Query: "{user_query}"

Create a task with the following JSON structure. Use the query and project context to fill in meaningful values:
{{
    "title": "Short task title (max 100 chars)",
    "description": "Detailed description of what needs to be done",
    "compliteTarget": "ISO 8601 datetime string for completion deadline (e.g., '2024-12-31T23:59:59Z')"
}}

Guidelines:
- title: Concise, action-oriented summary of the task
- description: Detailed explanation of what the task involves, based on the project context
- compliteTarget: A reasonable completion date (e.g., 1-4 weeks from now, adjust based on task complexity)

Return ONLY the JSON object, no other text."""

    try:
        messages = [{"role": "user", "content": prompt}]
        result_text = await _call_openrouter(messages, model="deepseek/deepseek-chat", max_tokens=1000, temperature=0.5)

        # Extract JSON from response
        json_match = re.search(r"\{[\s\S]*\}", result_text)
        if json_match:
            task_data = json.loads(json_match.group(0))
            return task_data
        else:
            logger.warning("Could not extract JSON from task generation: %s", result_text)
            return None
    except Exception as e:
        logger.exception("Error in task generation: %s", e)
        return None


async def _push_task_to_database(base: str, project_id: str, user_id: str, task_data: dict) -> dict | None:
    """
    Push task to database: POST {baseUrl}/api/v1/updateProject/:projectId/tasks
    """
    path = f"{base.rstrip('/')}/api/v1/updateProject/{project_id}/tasks"
    body = {
        "userId": user_id,
        "title": task_data.get("title", "Untitled Task"),
        "description": task_data.get("description", ""),
        "compliteTarget": task_data.get("compliteTarget", ""),
    }
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(path, json=body, timeout=10.0)
        if resp.status_code in (200, 201):
            return resp.json()
        logger.warning("Failed to push task to database: %s %s", resp.status_code, resp.text)
        return None
    except Exception as e:
        logger.exception("Exception pushing task to database: %s", e)
        return None


async def _generate_feedback_response(user_query: str, project_data: dict, last_message: str = "") -> str:
    """
    Generate a feedback/response message for user query without creating a task.
    Uses LLM to provide helpful response based on project context.
    Returns: feedback message string
    """
    prompt = f"""Provide helpful feedback and response to the user's query based on the project context.

Project Data:
{json.dumps(project_data, ensure_ascii=False, indent=2)}

Last conversation message:
{last_message.strip() if last_message and last_message.strip() else "(No previous message)"}

Current User Query: "{user_query}"

Provide a helpful, concise response to the user's question or message. Be practical and reference the project context where relevant.
Do not create a task, just provide helpful feedback and guidance based on their question."""

    try:
        messages = [{"role": "user", "content": prompt}]
        feedback = await _call_openrouter(messages, model="deepseek/deepseek-chat", max_tokens=1500, temperature=0.7)
        return feedback.strip()
    except Exception as e:
        logger.exception("Error in feedback generation: %s", e)
        return f"I encountered an error while processing your query: {str(e)}"


async def _generate_task_creation_response(user_query: str, task_data: dict, project_data: dict) -> str:
    """
    Generate a friendly response message confirming task creation.
    Uses LLM to provide a helpful confirmation message.
    Returns: response message string
    """
    prompt = f"""The user has just created a new task. Generate a friendly, professional response message confirming the task creation.

User Query: "{user_query}"

Created Task:
{json.dumps(task_data, ensure_ascii=False, indent=2)}

Project Context:
{json.dumps(project_data, ensure_ascii=False, indent=2)}

Generate a concise, friendly response message that:
1. Confirms the task has been created successfully
2. Highlights key details about the task (title and due date)
3. Offers brief encouragement or next steps
Keep it to 2-3 sentences."""

    try:
        messages = [{"role": "user", "content": prompt}]
        response = await _call_openrouter(messages, model="deepseek/deepseek-chat", max_tokens=500, temperature=0.7)
        return response.strip()
    except Exception as e:
        logger.exception("Error in task creation response generation: %s", e)
        task_title = task_data.get("title", "Task")
        return f"Task '{task_title}' has been created successfully!"


@router.post("/task_create/")
async def task_create_route(payload: TaskCreatePayload):
    """
    Create a task or provide feedback based on user query intent.
    - Get userId, projectId & user query from request
    - Fetch project data from DB
    - Get conversation from last_messege variable from task_ask route
    - Detect intent using DeepSeek LLM (general chat vs command for making task)
    - If intent is "create_task": generate task object, push to database, and provide response
    - If intent is "general_chat": provide feedback response instead of creating task
    """
    global last_messege
    
    base = os.getenv("PROJECT_SERVICE_URL")
    if not base:
        raise HTTPException(status_code=500, detail="PROJECT_SERVICE_URL not configured in environment")

    user_id = payload.userId
    project_id = payload.projectId
    user_query = (payload.query or "").strip()

    if not user_id or not project_id or not user_query:
        raise HTTPException(status_code=400, detail="userId, projectId, and query are required")

    # Fetch project data from DB
    project_data = await _fetch_project_data(base, project_id)
    if not project_data:
        raise HTTPException(status_code=500, detail="Failed to fetch project data from database. Check if projectId is valid or not.")

    # Get last message from task_ask route (handle empty case)
    last_message_text = last_messege.get("text") if last_messege else ""
    if not last_message_text:
        last_message_text = ""  # Ensure it's a string, not None

    # Detect intent using DeepSeek LLM
    intent_result = await _detect_task_intent_with_deepseek(user_query, last_message_text)
    intent = intent_result.get("intent", "general_chat")
    reason = intent_result.get("reason", "")

    logger.info(f"Task intent detection - Intent: {intent}, Reason: {reason}")

    # If intent is general_chat, provide feedback instead of creating task
    if intent == "general_chat":
        logger.info("User intent is general chat, generating feedback response")
        feedback = await _generate_feedback_response(user_query, project_data, last_message_text)
        return {
            "success": True,
            "intent": intent,
            "reason": reason,
            "task_created": False,
            "response": feedback,
        }

    # If intent is to create task, proceed with task creation
    logger.info("User intent is to create task, generating task object")
    
    # Use user query for task creation
    task_creation_query = user_query

    # Generate task object using LLM
    task_data = await _generate_task_from_query(task_creation_query, project_data, last_message_text)
    if not task_data:
        raise HTTPException(status_code=500, detail="Failed to generate task from query")

    # Push task to database
    response = await _push_task_to_database(base, project_id, user_id, task_data)
    if not response:
        raise HTTPException(status_code=500, detail="Failed to push task to database")

    # Generate a friendly response message for task creation
    task_response = await _generate_task_creation_response(user_query, task_data, project_data)

    return {
        "success": True,
        "intent": intent,
        "reason": reason,
        "task_created": True,
        "response": task_response,
        "task": task_data,
        "database_response": response,
    }





@router.post("/subtask_ask/")
async def subtask_ask_route(payload: SubtaskPayload):
    """
    Ask question about a subtask/task.
    - Fetch task/subtask data from DB
    - Save user message to chat history
    - Use LLM to answer based on task context
    - Save AI response to chat history
    - Store last message in global variable
    """
    global last_messege_subtask
    
    base = os.getenv("PROJECT_SERVICE_URL")
    if not base:
        raise HTTPException(status_code=500, detail="PROJECT_SERVICE_URL not configured in environment")

    user_id = payload.userId
    project_id = payload.projectId
    task_id = payload.taskId
    user_text = (payload.text or "").strip()

    if not user_id or not project_id or not task_id or not user_text:
        raise HTTPException(status_code=400, detail="userId, projectId, taskId, and text are required")

    # Store last message in global variable for subtask_create route
    last_messege_subtask = {
        "userId": user_id,
        "projectId": project_id,
        "taskId": task_id,
        "text": user_text,
    }

    # Save user message to chat history (using taskId as the projectOrTaskId)
    await _save_to_project_chat_history(base, user_id, task_id, user_text, is_ai=False, chat_type="ask")

    # Fetch task/subtask data from DB
    task_data = await _fetch_subtask_data(base, task_id, project_id)
    if not task_data:
        raise HTTPException(status_code=500, detail="Failed to fetch subtask data from database")

    # Build context with task data
    task_json = json.dumps(task_data, ensure_ascii=False, indent=2)
    context = (
        "You are a helpful task assistant. Answer the user's question about the following task/subtask. "
        "Provide concise and practical answers based on the task information.\n\n"
        f"TASK DATA:\n{task_json}\n\n"
        f"User Question: {user_text}\n\n"
        "Provide a helpful response."
    )

    # Call LLM to generate response
    reply = await generate_text_with_context(context, system_name="Task Assistant")

    # Save AI response to chat history (using taskId as the projectOrTaskId)
    await _save_to_project_chat_history(base, user_id, task_id, reply, is_ai=True, chat_type="ask")

    return {"user_text": user_text, "response": reply}


class SubtaskCreatePayload(BaseModel):
    userId: str
    projectId: str
    parentTaskId: str
    query: str


@router.post("/subtask_create/")
async def subtask_create_route(payload: SubtaskCreatePayload):
    """
    Create a subtask or provide feedback based on user query intent.
    - Get userId, projectId, parentTaskId & user query from request
    - Fetch parent task data from DB
    - Get conversation from last_messege_subtask variable
    - Detect intent using DeepSeek LLM (general chat vs command for making subtask)
    - If intent is "create_subtask": generate subtask object, push to database, and provide response
    - If intent is "general_chat": provide feedback response instead of creating subtask
    - Always provide a response message to the user
    """
    global last_messege_subtask, last_messege
    
    base = os.getenv("PROJECT_SERVICE_URL")
    if not base:
        raise HTTPException(status_code=500, detail="PROJECT_SERVICE_URL not configured in environment")

    user_id = payload.userId
    project_id = payload.projectId
    parent_task_id = payload.parentTaskId
    user_query = (payload.query or "").strip()

    if not user_id or not project_id or not parent_task_id or not user_query:
        raise HTTPException(status_code=400, detail="userId, projectId, parentTaskId, and query are required")

    # Fetch parent task data from DB
    parent_task_data = await _fetch_subtask_data(base, parent_task_id, project_id)
    if not parent_task_data:
        raise HTTPException(status_code=500, detail="Failed to fetch parent task data from database. Check if parentTaskId is valid or not.")

    # Get last message from subtask_ask route (handle empty case)
    last_subtask_message_text = last_messege_subtask.get("text") if last_messege_subtask else ""
    if not last_subtask_message_text:
        last_subtask_message_text = ""  # Ensure it's a string, not None

    # Get last message from task_ask route as fallback (handle empty case)
    last_task_message_text = last_messege.get("text") if last_messege else ""
    if not last_task_message_text:
        last_task_message_text = ""  # Ensure it's a string, not None

    # Detect intent using DeepSeek LLM
    intent_result = await _detect_subtask_intent_with_deepseek(user_query, last_subtask_message_text, last_task_message_text)
    intent = intent_result.get("intent", "general_chat")
    reason = intent_result.get("reason", "")

    logger.info(f"Subtask intent detection - Intent: {intent}, Reason: {reason}")

    # If intent is general_chat, provide feedback instead of creating subtask
    if intent == "general_chat":
        logger.info("User intent is general chat, generating feedback response")
        feedback = await _generate_subtask_feedback_response(user_query, parent_task_data, last_subtask_message_text)
        return {
            "success": True,
            "intent": intent,
            "reason": reason,
            "subtask_created": False,
            "response": feedback,
        }

    # If intent is to create subtask, proceed with subtask creation
    logger.info("User intent is to create subtask, generating subtask object")

    # Generate subtask object using LLM
    subtask_data = await _generate_subtask_from_query(user_query, parent_task_data, parent_task_id, project_id, user_id, last_subtask_message_text)
    if not subtask_data:
        raise HTTPException(status_code=500, detail="Failed to generate subtask from query")

    # Push subtask to database
    db_response = await _push_subtask_to_database(base, subtask_data)
    if not db_response:
        raise HTTPException(status_code=500, detail="Failed to push subtask to database")

    # Generate a friendly response message for subtask creation
    subtask_response = await _generate_subtask_creation_response(user_query, subtask_data, parent_task_data)

    return {
        "success": True,
        "intent": intent,
        "reason": reason,
        "subtask_created": True,
        "response": subtask_response,
        "subtask": subtask_data,
        "database_response": db_response,
    }

