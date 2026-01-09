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
        result_text = await _call_openrouter(messages, max_tokens=200, temperature=0.5)

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
        repaired = await _call_openrouter(messages, max_tokens=400, temperature=temp)
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

    # Fetch ask history for context
    history = await _fetch_history(base, user_id, chat_type="ask")
    texts = []
    for h in history:
        if isinstance(h, dict):
            t = h.get("text") or h.get("message") or ""
            texts.append(str(t))
        else:
            texts.append(str(h))
    aggregated = "\n".join(texts)

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
        "goal": "Build a Complete E-commerce Website",
        "tasks": [
            {
                "task": "Design Database Schema",
                "details": "Create MongoDB schemas for products, users, and orders",
                "taskDueDate": "2026-02-28",
                "subtasks": [
                    {"title": "Design User Schema", "subTaskDueDate": "2026-02-20"},
                    {"title": "Design Product Schema", "subTaskDueDate": "2026-02-23"},
                    {"title": "Design Order Schema", "subTaskDueDate": "2026-02-25"}
                ]
            },
            {
                "task": "Develop Frontend",
                "details": "Build React components for the website",
                "taskDueDate": "2026-03-15",
                "subtasks": [
                    {"title": "Create Home Page", "subTaskDueDate": "2026-03-05"},
                    {"title": "Create Product Listing Page", "subTaskDueDate": "2026-03-08"},
                    {"title": "Create Product Detail Page", "subTaskDueDate": "2026-03-12"}
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
    
    logger.info(f"LLM response for project creation: {gen[:300]}")

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

    # Send to createFullProject endpoint
    create_path = f"{base.rstrip('/')}/api/v1/project/createFullProject/{user_id}"
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(create_path, json=project_obj, timeout=10.0)
        if resp.status_code not in (200, 201):
            logger.warning("CreateFullProject failed: %s %s", resp.status_code, resp.text)
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

    return {"response": summary, "created": True, "project": project_obj}