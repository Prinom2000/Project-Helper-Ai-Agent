# api/v1/endpoints/project_task_question.py
import openai
import asyncio
from fastapi import APIRouter, HTTPException, Request
from app.schemas.project import Answer
from app.services.openai_service import generate_text
from app.utils.task_utils import extract_tasks
import os
import httpx
from dotenv import load_dotenv
load_dotenv()
from typing import Tuple, Optional, Dict, Any
from pydantic import BaseModel, Field
from fastapi import Body
import logging
import json
import re
import random
import asyncio
from datetime import datetime, timedelta
from typing import Optional, List
logging.basicConfig(level=logging.INFO)

router = APIRouter()

# In-memory storage for projects
projects = {}

# helpers
async def _fetch_project_by_id(project_id: str) -> Tuple[dict, bool, Optional[int]]:
    """
    Returns (normalized_project, external_fetched, numeric_index_if_in_memory)
    normalized_project has keys: goal, tasks, answered_questions
    """
    # numeric in-memory fallback
    if project_id.isdigit():
        idx = int(project_id)
        if idx in projects:
            return projects[idx], False, idx

    # fetch from external service
    base = os.getenv("PROJECT_SERVICE_URL")
    if not base:
        raise HTTPException(status_code=500, detail="PROJECT_SERVICE_URL not set in environment (.env)")
    get_path = f"{base.rstrip('/')}/api/v1/project/get/{project_id}/"
    async with httpx.AsyncClient() as client:
        resp = await client.get(get_path, timeout=10.0)
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=f"Could not fetch project: {resp.text}")
    payload = resp.json()
    project_data = payload.get("data") if isinstance(payload, dict) and payload.get("data") else payload
    project = {
        "goal": project_data.get("goal") or project_data.get("project_goal") or (project_data.get("_doc") or {}).get("goal"),
        "tasks": project_data.get("tasks", []),
        "answered_questions": project_data.get("answered_questions", []) or project_data.get("answeredQuestions", []),
    }
    return project, True, None

async def _persist_answered_questions_to_service(uid: str, answered_questions: list) -> tuple[bool, str]:
    """
    Try multiple endpoints, HTTP methods and payload shapes to persist answered_questions.
    Returns (success, debug_text) where debug_text contains last response or exception.
    """
    base = os.getenv("PROJECT_SERVICE_URL")
    if not base:
        return False, "PROJECT_SERVICE_URL not set in environment (.env)"
    candidate_paths = [
        f"{base.rstrip('/')}/api/v1/project/update/{uid}/",
        f"{base.rstrip('/')}/api/v1/project/patch/{uid}/",
        f"{base.rstrip('/')}/api/v1/project/{uid}/",
        f"{base.rstrip('/')}/api/v1/project/{uid}/update",
    ]
    # try common payload shapes
    payload_variants = [
        {"answered_questions": answered_questions},
        {"data": {"answered_questions": answered_questions}},
        {"project": {"answered_questions": answered_questions}},
    ]

    last_err = "no-attempt"
    async with httpx.AsyncClient() as client:
        for path in candidate_paths:
            for method in ("patch", "put", "post"):
                for payload in payload_variants:
                    try:
                        fn = getattr(client, method)
                        resp = await fn(path, json=payload, timeout=10.0)
                        debug = f"{method.upper()} {path} -> {resp.status_code} {resp.text[:1000]}"
                        logging.info("persist attempt: %s", debug)
                        if resp.status_code in (200, 201, 204):
                            return True, debug
                        last_err = debug
                    except Exception as e:
                        last_err = f"{method.upper()} {path} -> EXCEPTION: {e}"
                        logging.warning(last_err)
    return False, last_err

# Route to ask a question about the project goal
@router.post("/ask/{project_id}/")
async def ask_question(project_id: str):
    project, external_fetched, numeric_idx = await _fetch_project_by_id(project_id)

    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # generate next question
    if not project.get("answered_questions"):
        question = await generate_text(f"Based on this project goal, what is the question to ask generate only question no more word: {project['goal']}")
        new_entry = {"question": question, "answer": None}
    else:
        prev_qas = "\n".join(
            f"Q: {qa['question']}\nA: {qa.get('answer')}" for qa in project.get("answered_questions", [])
            if qa.get("answer") is not None
        )
        prompt = f"Project goal: {project['goal']}\n\nPrevious Q&A:\n{prev_qas}\n\nBased on the project goal and previous Q&A, what is the next question to ask?"
        question = await generate_text(prompt)
        new_entry = {"question": question, "answer": None}

    # save locally if in-memory
    if numeric_idx is not None:
        projects[numeric_idx].setdefault("answered_questions", []).append(new_entry)

    # persist to external service if fetched from there
    if external_fetched:
        qs = project.get("answered_questions", []) + [new_entry]
        success, debug = await _persist_answered_questions_to_service(project_id, qs)
        if not success:
            # return question but include persistence debug info to make troubleshooting easy from Postman
            return {"question": question}

    return {"question": question}

# Route to answer a question for the project
# @router.post("/answer_question/{project_id}/")
# async def answer_question(project_id: str, answer: Answer):
#     project, external_fetched, numeric_idx = await _fetch_project_by_id(project_id)

#     if not project:
#         raise HTTPException(status_code=404, detail="Project not found")

#     if not project.get("answered_questions"):
#         raise HTTPException(status_code=400, detail="No question to answer")

#     # store answer in the most recent question
#     project["answered_questions"][-1]["answer"] = answer.answer

#     # persist locally if in-memory
#     if numeric_idx is not None:
#         projects[numeric_idx]["answered_questions"] = project["answered_questions"]

#     # persist to external service when applicable
#     if external_fetched:
#         success = await _persist_answered_questions_to_service(project_id, project["answered_questions"])
#         if not success:
#             raise HTTPException(status_code=500, detail="Failed to persist answer to project service")

#     # return full project details (normalized)
#     return {"success": True, "project": project}

# New route for chat with project assistant
@router.post("/chat/{project_id}/")
async def chat_with_project_assistant(project_id: str, user_message: str):
    project, external_fetched, numeric_idx = await _fetch_project_by_id(project_id)

    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    project_goal = project["goal"]
    tasks = project.get("tasks", [])
    answered_questions = project.get("answered_questions", [])

    context = f"Project Goal: {project_goal}\n\n"
    context += "Tasks:\n"
    for task in tasks:
        # support both dict task format and simple strings
        if isinstance(task, dict):
            context += f"- {task.get('task')}\n"
            for subtask in task.get("subtasks", []):
                context += f"  - Subtask: {subtask}\n"
            if 'details' in task and task['details']:
                context += f"  - Details: {task['details']}\n"
        else:
            context += f"- {task}\n"

    context += "\nAnswered Questions:\n"
    for q_a in answered_questions:
        context += f"Q: {q_a.get('question')}\nA: {q_a.get('answer')}\n"

    context += f"\nUser's message: {user_message}\n\nAssistant, based on this information, answer the user's query."

    response = await generate_text_with_context(context)
    # return only the assistant response (remove project_id and project payload)
    return {"response": response}

# Route to get project details
@router.get("/get_project/{project_id}/")
async def get_project(project_id: str):
    project, external_fetched, numeric_idx = await _fetch_project_by_id(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return {"project": project}

# reuse existing helper to call OpenAI with context
async def generate_text_with_context(context: str):
    try:
        response = await asyncio.to_thread(openai.ChatCompletion.create,
                                            model="gpt-3.5-turbo",
                                            messages=[
                                                {"role": "system", "content": "You are a helpful assistant named OLLIE that helps users with project details."},
                                                {"role": "user", "content": context}
                                            ],
                                            max_tokens=300,
                                            temperature=0.7,
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

class TitleRequest(BaseModel):
    # Python attribute has no trailing space, but incoming JSON key can be "user_text "
    user_text: str = Field(..., alias="user_text ")
    class Config:
        allow_population_by_field_name = True
        extra = "forbid"

@router.post("/generate_title/")
async def generate_title(payload: TitleRequest = Body(...)):
    """
    Accepts JSON like:
    { "user_text ": "your long project idea or short title" }
    Returns a short, single-line project title (max 6 words).
    """
    text = (payload.user_text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="No project idea text provided")

    prompt = (
        "Summarize the following project title/idea into a short, clear project title (max 6 words). "
        "Return only the title on a single line with no additional explanation.\n\n"
        f"{text}"
    )

    title = (await generate_text(prompt)) or ""
    title = title.strip().splitlines()[0].strip(' "\'')
    parts = title.split()
    if len(parts) > 6:
        title = " ".join(parts[:6])

    return {"title": title}

@router.get("/project_tasks/{project_id}/")
async def project_tasks(project_id: str, prompt: Optional[str] = None):
    """
    Return normalized tasks for a project (task objects with optional subtasks).
    - uid (project_id) is required
    - prompt is optional: if provided, use it to create/modify tasks based on natural language
      Example: "make a task named call mom and deadline will be after 3 days"
    Response: { "project_id": "...", "tasks": [ { "task": "...", "details":"...", "datetime":"...", "subtasks": [...] } ] }
    """
    project, external_fetched, numeric_idx = await _fetch_project_by_id(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # If prompt is provided, parse it to extract custom task creation instructions
    custom_task = None
    if prompt and prompt.strip():
        custom_task = await _parse_prompt_for_task(prompt, project.get("goal", ""))
    
    def _normalize_task_item(item):
        # item can be dict or string
        if isinstance(item, dict):
            title = item.get("task") or item.get("title") or item.get("name") or str(item)
            details = item.get("details") or item.get("description") or ""
            dt = item.get("datetime") or item.get("date") or item.get("due") or ""
            subtasks = item.get("subtasks") or item.get("sub_tasks") or item.get("children") or []
            normalized_subs = []
            for s in subtasks or []:
                if isinstance(s, dict):
                    sub_title = s.get("subtask") or s.get("task") or s.get("title") or ""
                    sub_details = s.get("details") or s.get("description") or ""
                    sub_dt = s.get("datetime") or s.get("date") or s.get("due") or ""

                    parsed_inner = None
                    if isinstance(sub_title, str):
                        parsed_inner = _parse_possible_json_string(sub_title)
                    if parsed_inner and isinstance(parsed_inner, dict):
                        sub_title = parsed_inner.get("subtask") or parsed_inner.get("task") or parsed_inner.get("title") or sub_title
                        sub_details = parsed_inner.get("details") or sub_details
                        sub_dt = parsed_inner.get("datetime") or parsed_inner.get("date") or parsed_inner.get("due") or sub_dt
                    else:
                        parsed_details = _parse_possible_json_string(sub_details) if isinstance(sub_details, str) else None
                        if parsed_details and isinstance(parsed_details, dict):
                            sub_title = parsed_details.get("subtask") or parsed_details.get("task") or sub_title
                            sub_details = parsed_details.get("details") or sub_details
                            sub_dt = parsed_details.get("datetime") or sub_dt

                    normalized_subs.append({"subtask": sub_title, "details": sub_details, "datetime": sub_dt})
                else:
                    parsed = _parse_possible_json_string(str(s))
                    if isinstance(parsed, dict):
                        sub_title = parsed.get("subtask") or parsed.get("task") or parsed.get("title") or str(parsed)
                        sub_details = parsed.get("details") or parsed.get("description") or ""
                        sub_dt = parsed.get("datetime") or parsed.get("date") or parsed.get("due") or ""
                        normalized_subs.append({"subtask": sub_title, "details": sub_details, "datetime": sub_dt})
                    else:
                        normalized_subs.append({"subtask": str(s), "details": "", "datetime": ""})
            return {"task": title, "details": details, "datetime": dt, "subtasks": normalized_subs}
        else:
            return {"task": str(item), "details": "", "datetime": "", "subtasks": []}

    raw_tasks = project.get("tasks") or []
    normalized = [_normalize_task_item(t) for t in raw_tasks if t is not None]

    # If custom task from prompt, add it to the tasks
    if custom_task:
        normalized.append(custom_task)

    # if we already have tasks, ensure they have subtasks
    if normalized:
        await _ensure_subtasks_for_tasks(normalized, project.get("goal", ""), project.get("answered_questions", []), prompt=prompt)
        _ensure_subtasks_before_task_datetimes(normalized)
        return {"project_id": project_id, "tasks": normalized}

    # try using local extractor if available
    try:
        generated = extract_tasks(project.get("goal", ""), project.get("answered_questions", []))
        if generated:
            normalized = [_normalize_task_item(t) for t in generated]
            if custom_task:
                normalized.append(custom_task)
            await _ensure_subtasks_for_tasks(normalized, project.get("goal", ""), project.get("answered_questions", []), prompt=prompt)
            _ensure_subtasks_before_task_datetimes(normalized)
            return {"project_id": project_id, "tasks": normalized}
    except Exception:
        pass

    # fallback: ask assistant to produce a JSON array of tasks with optional subtasks
    goal = project.get("goal", "") or ""
    answered = project.get("answered_questions", []) or []
    answered_text = "\n".join(f"Q: {q.get('question')}\nA: {q.get('answer')}" for q in answered if q)
    llm_prompt = (
        "Produce a JSON array of tasks for this project. Each element must be an object with keys:\n"
        "  - task (string)\n"
        "  - details (string)\n"
        "  - datetime (ISO 8601 string)\n"
        "  - subtasks (array of objects)  // each subtask object: { subtask: string, details: string, datetime: ISO }\n\n"
        "Do NOT include any extra text, only the JSON array. Keep tasks concise.\n\n"
        f"Project goal: {goal}\n\n"
        f"Answered questions:\n{answered_text}\n\n"
        + (f"User prompt: {prompt}\n\n" if prompt else "")
        + "Return JSON array now."
    )

    try:
        gen_text = await generate_text_with_context(llm_prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate tasks: {e}")

    # try parse JSON
    try:
        def _try_parse_gen_text(text: str):
            cleaned = re.sub(r"```(?:json)?\s*", "", text, flags=re.IGNORECASE)
            cleaned = cleaned.replace("```", "")

            m = re.search(r"\[.*\]", cleaned, flags=re.DOTALL)
            if m:
                candidate = m.group(0)
                try:
                    parsed = json.loads(candidate)
                    if isinstance(parsed, list):
                        return parsed
                except Exception:
                    fixed = re.sub(r",\s*}", "}", candidate)
                    fixed = re.sub(r",\s*\]", "]", fixed)
                    try:
                        parsed = json.loads(fixed)
                        if isinstance(parsed, list):
                            return parsed
                    except Exception:
                        pass

            task_vals = re.findall(r'"task"\s*:\s*"([^"]+)"', cleaned) + re.findall(r"'task'\s*:\s*'([^']+)'", cleaned)
            if task_vals:
                return [{"task": t, "subtasks": []} for t in task_vals]

            items = []
            for line in cleaned.splitlines():
                s = line.strip()
                if not s:
                    continue
                if re.match(r'^[\[\]\{\}\s,`]+$', s):
                    continue
                m = re.search(r'"task"\s*:\s*"([^"]+)"', s)
                if m:
                    items.append({"task": m.group(1), "subtasks": []})
                    continue
                if re.match(r'^"(subtasks|sub_tasks|children)"', s, flags=re.IGNORECASE):
                    continue
                s = re.sub(r'^[\-\•\*\s]+', '', s)
                s = s.strip('",')
                if len(s) < 2:
                    continue
                items.append({"task": s, "subtasks": []})
            return items

        parsed = _try_parse_gen_text(gen_text)
        if isinstance(parsed, list) and parsed:
            normalized = []
            for el in parsed:
                if isinstance(el, dict):
                    normalized.append(_normalize_task_item(el))
                elif isinstance(el, str):
                    parsed_sub = _parse_possible_json_string(el)
                    if isinstance(parsed_sub, dict):
                        normalized.append(_normalize_task_item(parsed_sub))
                    else:
                        normalized.append(_normalize_task_item({"task": el, "subtasks": []}))
            if custom_task:
                normalized.append(custom_task)
            await _ensure_subtasks_for_tasks(normalized, project.get("goal", ""), project.get("answered_questions", []), prompt=prompt)
            _ensure_subtasks_before_task_datetimes(normalized)
            return {"project_id": project_id, "tasks": normalized}
    except Exception:
        logging.exception("Failed parsing generated tasks text")
        lines = []
        for l in gen_text.splitlines():
            s = l.strip()
            if not s or re.match(r'^[\[\]\{\}\s,`]+$', s):
                continue
            s = re.sub(r'^[\-\•\*\s]+', '', s)
            s = s.strip('",')
            if len(s) >= 2:
                lines.append(s)
        if lines:
            normalized = [{"task": l, "subtasks": []} for l in lines[:50]]
            if custom_task:
                normalized.append(custom_task)
            return {"project_id": project_id, "tasks": normalized, "raw": gen_text[:2000]}

    # last resort
    if custom_task:
        return {"project_id": project_id, "tasks": [custom_task]}
    return {"project_id": project_id, "tasks": []}


async def _parse_prompt_for_task(prompt: str, goal: str) -> Optional[dict]:
    """
    Parse natural language prompt to extract task creation instructions.
    Examples:
    - "make a task named call mom and deadline will be after 3 days"
    - "create task: buy groceries, due in 5 days"
    - "add task call doctor tomorrow"
    
    Returns a task dict: { task, details, datetime, subtasks }
    """
    prompt_lower = prompt.lower().strip()
    
    # Check if this is a task creation prompt
    task_keywords = ["make a task", "create task", "add task", "new task", "task named", "task called"]
    is_task_creation = any(keyword in prompt_lower for keyword in task_keywords)
    
    if not is_task_creation:
        # Not a task creation prompt, return None
        return None
    
    # Use LLM to parse the prompt intelligently
    parse_prompt = (
        f"Extract task information from this user request: \"{prompt}\"\n\n"
        "Return ONLY a JSON object with these fields:\n"
        "- task: the task name/title\n"
        "- details: brief description (can be empty string)\n"
        "- days_from_now: number of days from today for the deadline (integer)\n\n"
        "Examples:\n"
        "Input: 'make a task named call mom and deadline will be after 3 days'\n"
        "Output: {\"task\":\"Call mom\",\"details\":\"Contact mother\",\"days_from_now\":3}\n\n"
        "Input: 'create task buy groceries due in 5 days'\n"
        "Output: {\"task\":\"Buy groceries\",\"details\":\"Purchase grocery items\",\"days_from_now\":5}\n\n"
        "Input: 'add task call doctor tomorrow'\n"
        "Output: {\"task\":\"Call doctor\",\"details\":\"Schedule appointment\",\"days_from_now\":1}\n\n"
        "Return only the JSON object, no other text."
    )
    
    try:
        response = await generate_text_with_context(parse_prompt)
        cleaned = re.sub(r'```(?:json)?\s*', '', response, flags=re.IGNORECASE).replace('```', '').strip()
        
        # Extract JSON object
        match = re.search(r'\{.*\}', cleaned, flags=re.DOTALL)
        if match:
            parsed = json.loads(match.group(0))
            
            task_name = parsed.get("task", "").strip()
            details = parsed.get("details", "").strip()
            days_from_now = parsed.get("days_from_now", 1)
            
            if not task_name:
                return None
            
            # Calculate datetime
            now = _now_utc()
            deadline = now + timedelta(days=int(days_from_now))
            deadline = deadline.replace(hour=9, minute=0, second=0, microsecond=0)
            datetime_iso = deadline.isoformat() + "Z"
            
            return {
                "task": task_name,
                "details": details if details else f"Complete: {task_name}",
                "datetime": datetime_iso,
                "subtasks": []
            }
    except Exception as e:
        logging.warning(f"Failed to parse prompt for task: {e}")
    
    # Fallback: simple regex parsing
    try:
        # Extract task name
        task_match = re.search(r'(?:task|named|called)\s+([^,\.]+?)(?:\s+(?:and|with|due|deadline|after|in)|$)', prompt_lower)
        task_name = task_match.group(1).strip() if task_match else "New Task"
        task_name = task_name.title()
        
        # Extract days
        days_match = re.search(r'(?:after|in)\s+(\d+)\s+days?', prompt_lower)
        if days_match:
            days = int(days_match.group(1))
        elif 'tomorrow' in prompt_lower:
            days = 1
        elif 'today' in prompt_lower:
            days = 0
        else:
            days = 1
        
        now = _now_utc()
        deadline = now + timedelta(days=days)
        deadline = deadline.replace(hour=9, minute=0, second=0, microsecond=0)
        datetime_iso = deadline.isoformat() + "Z"
        
        return {
            "task": task_name,
            "details": f"Complete: {task_name}",
            "datetime": datetime_iso,
            "subtasks": []
        }
    except Exception as e:
        logging.warning(f"Fallback parsing failed: {e}")
        return None


async def _generate_subtasks_for_task(task_title: str, goal: str, answered_questions: list) -> list:
    """
    Ask the assistant to produce between 2 and 5 concise subtasks (random count) for a given task.
    Returns a list of subtask dicts: { subtask: str, details: str, datetime: ISO }
    """
    n = random.randint(2, 5)
    answered_text = "\n".join(f"Q: {q.get('question')}\nA: {q.get('answer')}" for q in (answered_questions or []) if q)
    prompt = (
        f"Project goal: {goal}\n\nAnswered questions:\n{answered_text}\n\n"
        f"For the task: \"{task_title}\", produce between 2 and {n} concise subtasks that help complete the task. "
        "For each subtask include a short 'details' description and a suggested ISO-8601 datetime (deadline). "
        "Return ONLY a valid JSON array with no extra text. Each object must have exactly these keys: subtask, details, datetime.\n"
        "Example format: [{\"subtask\":\"Do something\",\"details\":\"Brief description\",\"datetime\":\"2025-10-23T10:00:00Z\"}]\n"
        "Do not include any markdown, code blocks, or explanatory text."
    )
    try:
        gen = await generate_text_with_context(prompt)
    except Exception:
        return []
 
    cleaned = gen.strip()
    cleaned = re.sub(r'```(?:json)?\s*', '', cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace('```', '')
    
    match = re.search(r'\[.*\]', cleaned, flags=re.DOTALL)
    if match:
        cleaned = match.group(0)
    
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list) and parsed:
            subs = []
            for el in parsed:
                if isinstance(el, dict):
                    sub_title = str(el.get("subtask", "")).strip()
                    sub_details = str(el.get("details", "")).strip()
                    sub_dt = str(el.get("datetime", "")).strip()
                    
                    sub_title = re.sub(r'^(subtask|details|datetime)["\s]*:\s*["\s]*', '', sub_title)
                    sub_details = re.sub(r'^(subtask|details|datetime)["\s]*:\s*["\s]*', '', sub_details)
                    sub_dt = re.sub(r'^(subtask|details|datetime)["\s]*:\s*["\s]*', '', sub_dt)
                    
                    if sub_title and len(sub_title) > 1:
                        subs.append({
                            "subtask": sub_title,
                            "details": sub_details if sub_details else f"Perform: {sub_title}",
                            "datetime": sub_dt if sub_dt else _random_future_datetime_iso()
                        })
                elif isinstance(el, str) and el.strip():
                    subs.append({
                        "subtask": el.strip(),
                        "details": f"Perform: {el.strip()}",
                        "datetime": _random_future_datetime_iso()
                    })
            
            if subs:
                return subs[:n]
    except json.JSONDecodeError:
        try:
            fixed = re.sub(r',\s*}', '}', cleaned)
            fixed = re.sub(r',\s*\]', ']', fixed)
            parsed = json.loads(fixed)
            if isinstance(parsed, list) and parsed:
                subs = []
                for el in parsed:
                    if isinstance(el, dict):
                        sub_title = str(el.get("subtask", "")).strip()
                        sub_details = str(el.get("details", "")).strip()
                        sub_dt = str(el.get("datetime", "")).strip()
                        
                        sub_title = re.sub(r'^(subtask|details|datetime)["\s]*:\s*["\s]*', '', sub_title)
                        sub_details = re.sub(r'^(subtask|details|datetime)["\s]*:\s*["\s]*', '', sub_details)
                        sub_dt = re.sub(r'^(subtask|details|datetime)["\s]*:\s*["\s]*', '', sub_dt)
                        
                        if sub_title and len(sub_title) > 1:
                            subs.append({
                                "subtask": sub_title,
                                "details": sub_details if sub_details else f"Perform: {sub_title}",
                                "datetime": sub_dt if sub_dt else _random_future_datetime_iso()
                            })
                if subs:
                    return subs[:n]
        except Exception:
            pass
    except Exception:
        pass

    result = []
    subtask_pattern = r'"subtask"\s*:\s*"([^"]+)"'
    details_pattern = r'"details"\s*:\s*"([^"]+)"'
    datetime_pattern = r'"datetime"\s*:\s*"([^"]+)"'
    
    subtask_matches = re.findall(subtask_pattern, cleaned)
    details_matches = re.findall(details_pattern, cleaned)
    datetime_matches = re.findall(datetime_pattern, cleaned)
    
    for i in range(min(len(subtask_matches), n)):
        sub_title = subtask_matches[i].strip()
        sub_title = re.sub(r'^(subtask|details|datetime)["\s]*:\s*["\s]*', '', sub_title)
        
        if len(sub_title) > 1:
            sub_details = details_matches[i].strip() if i < len(details_matches) else f"Perform: {sub_title}"
            sub_details = re.sub(r'^(subtask|details|datetime)["\s]*:\s*["\s]*', '', sub_details)
            
            sub_dt = datetime_matches[i].strip() if i < len(datetime_matches) else _random_future_datetime_iso()
            sub_dt = re.sub(r'^(subtask|details|datetime)["\s]*:\s*["\s]*', '', sub_dt)
            
            result.append({
                "subtask": sub_title,
                "details": sub_details if sub_details else f"Perform: {sub_title}",
                "datetime": sub_dt if sub_dt else _random_future_datetime_iso()
            })
    
    if result:
        return result
    
    generic_subtasks = [
        f"Prepare for {task_title}",
        f"Execute {task_title}",
        f"Complete {task_title}"
    ]
    
    return [{
        "subtask": generic_subtasks[i % len(generic_subtasks)],
        "details": f"Perform: {generic_subtasks[i % len(generic_subtasks)]}",
        "datetime": _random_future_datetime_iso()
    } for i in range(min(n, 3))]

async def _ensure_subtasks_for_tasks(tasks: list, goal: str, answered_questions: list, prompt: Optional[str] = None):
    """
    Ensure each task dict has 'subtasks' populated (possibly empty list).
    Will call LLM for tasks with no subtasks and also ensure tasks have details and datetime.
    """
    # collect tasks that need subtasks
    to_generate = []
    for idx, t in enumerate(tasks):
        if not isinstance(t, dict):
            # normalize non-dict to dict
            tasks[idx] = {"task": str(t), "details": "", "datetime": "", "subtasks": []}
            t = tasks[idx]
        t.setdefault("subtasks", [])
        t.setdefault("details", "")
        t.setdefault("datetime", "")
        if not t["subtasks"]:
            to_generate.append((idx, t["task"]))
        # if details or datetime missing, we'll also fill them below
    if not to_generate:
        # still ensure missing details/datetimes are filled
        for idx, t in enumerate(tasks):
            if not t.get("details") or not t.get("datetime"):
                meta = await _generate_task_meta(t["task"], goal, answered_questions, prompt)
                tasks[idx]["details"] = t.get("details") or meta.get("details", "")
                tasks[idx]["datetime"] = t.get("datetime") or meta.get("datetime", "")
                # ensure datetime returned (or existing) is after now
                tasks[idx]["datetime"] = _ensure_datetime_after_now_iso(t.get("datetime") or meta.get("datetime", ""))
        return
    # generate sequentially to avoid hitting rate-limits; can be parallelized with care
    for idx, title in to_generate:
        try:
            subs = await _generate_subtasks_for_task(title, goal, answered_questions)
            tasks[idx]["subtasks"] = subs or []
        except Exception:
            tasks[idx]["subtasks"] = []
        # ensure task-level details/datetime
        if not tasks[idx].get("details") or not tasks[idx].get("datetime"):
            meta = await _generate_task_meta(title, goal, answered_questions, prompt)
            tasks[idx]["details"] = tasks[idx].get("details") or meta.get("details", "")
            # ensure assigned datetime is after now
            tasks[idx]["datetime"] = _ensure_datetime_after_now_iso(tasks[idx].get("datetime") or meta.get("datetime", ""))

    # final pass: ensure every task datetime is after now
    for t in tasks:
        t["datetime"] = _ensure_datetime_after_now_iso(t.get("datetime", ""))
    return

# --- new / updated datetime helpers ---
def _parse_iso_like(s: str) -> Optional[datetime]:
    if not s or not isinstance(s, str):
        return None
    txt = s.strip()
    # remove trailing Z if present
    if txt.endswith("Z"):
        txt = txt[:-1]
    try:
        return datetime.fromisoformat(txt)
    except Exception:
        try:
            return datetime.strptime(txt, "%Y-%m-%d")
        except Exception:
            try:
                return datetime.strptime(txt, "%d/%m/%Y")
            except Exception:
                return None

def _now_utc() -> datetime:
    return datetime.utcnow().replace(microsecond=0)

def _ensure_datetime_after_now_iso(dt_iso: str, default_days_ahead: int = 1) -> str:
    """
    Ensure dt_iso is an ISO datetime string after current time.
    If dt_iso is empty/invalid or <= now, returns now + default_days_ahead at 09:00 UTC.
    Returns ISO string ending with Z.
    """
    now = _now_utc()
    parsed = _parse_iso_like(dt_iso or "")
    if parsed is None or parsed <= now:
        # choose a reasonable default future datetime (09:00)
        dt = (now + timedelta(days=default_days_ahead)).replace(hour=9, minute=0, second=0)
    else:
        dt = parsed
    return dt.replace(second=0, microsecond=0).isoformat() + "Z"

def _random_future_datetime_iso(min_days: int = 1, max_days: int = 14) -> str:
    """
    Generate a future datetime (ISO Z) strictly after now.
    """
    now = _now_utc()
    days = random.randint(min_days, max_days)
    hour = random.randint(9, 18)
    minute = random.choice([0, 15, 30, 45])
    dt = (now + timedelta(days=days)).replace(hour=hour, minute=minute, second=0, microsecond=0)
    if dt <= now:
        dt = now + timedelta(days=1, hours=1)
        dt = dt.replace(minute=0, second=0, microsecond=0)
    return dt.isoformat() + "Z"

def _random_before_datetime_iso(target_iso: str, min_hours: int = 1, max_days_before: int = 7) -> str:
    """
    Return an ISO datetime string strictly before target_iso and after now.
    If target_iso is invalid or <= now, target is set to now + 1 day.
    """
    now = _now_utc()
    tgt = _parse_iso_like(target_iso or "")
    if not tgt or tgt <= now:
        tgt = now + timedelta(days=1)
    # compute a random delta between min_hours and max_days_before
    max_seconds = max_days_before * 24 * 3600
    min_seconds = max(min_hours * 3600, 3600)
    rand_seconds = random.randint(min_seconds, max_seconds) if max_seconds > min_seconds else min_seconds
    sub_dt = tgt - timedelta(seconds=rand_seconds)
    if sub_dt <= now:
        sub_dt = now + timedelta(hours=1)
    return sub_dt.replace(second=0, microsecond=0).isoformat() + "Z"

def _ensure_subtasks_before_task_datetimes(tasks: List[dict]):
    """
    Ensure subtask datetimes are strictly before parent task datetime, and all datetimes are after now.
    Adjusts datetimes in-place.
    """
    now = _now_utc()
    for t in tasks:
        task_dt_iso = t.get("datetime") or ""
        # ensure task datetime exists and is after now
        t["datetime"] = _ensure_datetime_after_now_iso(task_dt_iso, default_days_ahead=1)
        try:
            task_dt = _parse_iso_like(t["datetime"])
        except Exception:
            task_dt = None
        if not task_dt:
            task_dt = now + timedelta(days=1)
            t["datetime"] = task_dt.replace(hour=9, minute=0, second=0, microsecond=0).isoformat() + "Z"

        for sub in t.get("subtasks", []):
            sub_dt_iso = sub.get("datetime") or ""
            sub_parsed = _parse_iso_like(sub_dt_iso) if sub_dt_iso else None
            # if subtask datetime missing or invalid, generate one before task_dt
            if not sub_parsed:
                sub["datetime"] = _random_before_datetime_iso(task_dt.isoformat())
            else:
                # if subtask is not before task, move it
                if sub_parsed >= task_dt:
                    sub["datetime"] = _random_before_datetime_iso(task_dt.isoformat())
                else:
                    # still ensure subtask is after now
                    if sub_parsed <= now:
                        sub["datetime"] = _random_before_datetime_iso(task_dt.isoformat())
                    else:
                        # normalize format
                        sub["datetime"] = sub_parsed.replace(second=0, microsecond=0).isoformat() + "Z"

async def _generate_task_meta(task_title: str, goal: str, answered_questions: list, prompt: Optional[str] = None) -> dict:
    """
    Generate a small metadata object for a task: { details, datetime }.
    Fallbacks to simple autogenerated text and random datetime.
    """
    answered_text = "\n".join(f"Q: {q.get('question')}\nA: {q.get('answer')}" for q in (answered_questions or []) if q)
    p = (
        f"Project goal: {goal}\n\nAnswered questions:\n{answered_text}\n\n"
        + (f"User prompt: {prompt}\n\n" if prompt else "")
        + f"For the task \"{task_title}\", produce a JSON object: {{\"details\":\"short action-oriented details\",\"datetime\":\"ISO-8601 datetime\"}}. Return only JSON."
    )
    try:
        gen = await generate_text_with_context(p)
        parsed = json.loads(re.sub(r"```(?:json)?\s*", "", (gen or ""), flags=re.IGNORECASE).replace("```", ""))
        if isinstance(parsed, dict):
            return {"details": parsed.get("details", "").strip(), "datetime": parsed.get("datetime", "").strip()}
    except Exception:
        pass
    # fallback simple
    return {"details": f"Complete task: {task_title}", "datetime": _random_future_datetime_iso()}

# helper: try to parse a string that may contain a JSON object or key/value fragments
def _parse_possible_json_string(s: str) -> Optional[dict]:
    if not s or not isinstance(s, str):
        return None
    txt = s.strip()
    # remove markdown fences and surrounding quotes
    txt = re.sub(r"^```(?:json)?\s*", "", txt, flags=re.IGNORECASE).replace("```", "")
    txt = txt.strip()

    # If the whole value is quoted (e.g. "\"{...}\"" or "'{...}'"), strip outer quotes
    if (txt.startswith('"') and txt.endswith('"')) or (txt.startswith("'") and txt.endswith("'")):
        try:
            txt = txt[1:-1]
        except Exception:
            pass
    txt = txt.strip()

    # try multiple normalization attempts to handle escaped JSON strings
    candidates = [txt]

    # replace escaped quotes
    if '\\"' in txt:
        candidates.append(txt.replace('\\"', '"'))

    # try unicode_escape decode (handles \u and other escapes)
    try:
        candidates.append(txt.encode('utf-8').decode('unicode_escape'))
    except Exception:
        pass

    # try removing surrounding backslashes if present
    if txt.startswith('\\') and txt.endswith('\\'):
        candidates.append(txt.strip('\\'))

    for cand in candidates:
        cand = cand.strip()
        if not cand:
            continue
        # attempt JSON parse
        try:
            parsed = json.loads(cand)
            return parsed
        except Exception:
            # try to normalize fancy quotes then parse
            try:
                fixed = cand.replace("“", "\"").replace("”", "\"").replace("’", "'").replace("‘", "'")
                parsed = json.loads(fixed)
                return parsed
            except Exception:
                pass

    # try to extract a JSON object substring
    m = re.search(r"\{.*\}", txt, flags=re.DOTALL)
    if m:
        sub = m.group(0)
        try:
            return json.loads(sub)
        except Exception:
            # try to unescape and parse
            try:
                fixed = sub.replace('\\"', '"').encode('utf-8').decode('unicode_escape')
                return json.loads(fixed)
            except Exception:
                pass

    # try to extract key/value pairs like "subtask":"..." etc.
    kv = {}
    for key in ("subtask", "task", "details", "datetime", "date", "due"):
        m = re.search(rf'"{key}"\s*:\s*"([^"]+)"', txt)
        if m:
            kv[key] = m.group(1)
        else:
            m2 = re.search(rf"'{key}'\s*:\s*'([^']+)'", txt)
            if m2:
                kv[key] = m2.group(1)
    if kv:
        return kv
    return None

class TaskEditRequest(BaseModel):
    task_id: str
    prompt: str
    project_id: str  # needed to locate the task



############################################################
async def _fetch_task_by_id(project_id: str, task_id: str) -> Tuple[Optional[dict], bool]:
    """
    Fetch a single task by project_id and task_id.
    Returns (task_dict, success) where task_dict contains: task, details, datetime, subtasks
    
    The function tries:
    1. Fetch from external service using getSingletask endpoint
    2. Fall back to in-memory storage if project_id is numeric
    3. Fall back to fetching full project and extracting the task
    """
    
    # Try fetching from external service first
    base = os.getenv("PROJECT_SERVICE_URL")
    if base:
        # Try the getSingletask endpoint
        get_task_path = f"{base.rstrip('/')}/api/v1/project/getSingletask/{project_id}/{task_id}/"
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(get_task_path, timeout=10.0)
            
            if resp.status_code == 200:
                payload = resp.json()
                # Handle different response structures
                task_data = payload.get("data") if isinstance(payload, dict) and payload.get("data") else payload
                
                # If task_data is the task itself
                if isinstance(task_data, dict) and "task" in task_data:
                    return task_data, True
                
                # If task_data contains a task field
                if isinstance(task_data, dict) and task_data.get("task"):
                    return task_data.get("task"), True
                    
        except Exception as e:
            logging.warning(f"Failed to fetch task from external service: {e}")
    
    # Fallback 1: Check in-memory storage if project_id is numeric
    if project_id.isdigit():
        idx = int(project_id)
        if idx in projects:
            project = projects[idx]
            tasks = project.get("tasks", [])
            
            # Try to find task by task_id (could be index or ID)
            if task_id.isdigit():
                task_idx = int(task_id)
                if 0 <= task_idx < len(tasks):
                    task = tasks[task_idx]
                    if isinstance(task, dict):
                        return task, True
                    return {"task": str(task), "details": "", "datetime": "", "subtasks": []}, True
            
            # Try to find task by matching ID field
            for task in tasks:
                if isinstance(task, dict):
                    if task.get("id") == task_id or task.get("_id") == task_id:
                        return task, True
    
    # Fallback 2: Fetch full project and extract the task
    try:
        project, external_fetched, numeric_idx = await _fetch_project_by_id(project_id)
        
        if project:
            tasks = project.get("tasks", [])
            
            # Try to find by index
            if task_id.isdigit():
                task_idx = int(task_id)
                if 0 <= task_idx < len(tasks):
                    task = tasks[task_idx]
                    if isinstance(task, dict):
                        return task, True
                    return {"task": str(task), "details": "", "datetime": "", "subtasks": []}, True
            
            # Try to find by ID field
            for task in tasks:
                if isinstance(task, dict):
                    if task.get("id") == task_id or task.get("_id") == task_id:
                        return task, True
    
    except Exception as e:
        logging.error(f"Failed to fetch project for task lookup: {e}")
    
    # Task not found
    return None, False

############################################################ to get a single task by id or index
@router.get("/task/{project_id}/{task_id}")
async def get_task(project_id: str, task_id: str):
    """
    Get details of a specific task by project_id and task_id using backend's getSingletask API.
    The task_id can be either the task's unique ID or its index in the tasks array.
    """
    task, success = await _fetch_task_by_id(project_id, task_id)
    
    if not success or not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    # Ensure datetime is valid and after now
    task["datetime"] = _ensure_datetime_after_now_iso(task.get("datetime", ""))
    
    # Ensure all subtask datetimes are valid
    for subtask in task.get("subtasks", []):
        if isinstance(subtask, dict):
            subtask["datetime"] = _ensure_datetime_after_now_iso(subtask.get("datetime", ""))
    
    return {
        "success": True,
        "task": task,
        "project_id": project_id
    }

############################################################ for edit task
class TaskEditPromptRequest(BaseModel):
    prompt: str
    project_id: str  # needed to persist changes back

# Also update the edit endpoint to log the response
@router.patch("/task/{task_id}/edit")
async def edit_task(task_id: str, request: TaskEditPromptRequest):
    """
    Edit a task based on natural language prompt.
    
    Supported operations:
    - Edit task name: "change task name to 'Buy groceries'"
    - Edit task details: "update details to 'purchase items from store'"
    - Edit task datetime: "change deadline to 3 days from now" or "set date to 2025-11-15"
    - Add subtask: "add subtask 'call supplier'"
    - Edit subtask: "change first subtask name to 'email supplier'"
    - Replace subtask: "replace second subtask with 'review documents'"
    - Remove subtask: "remove last subtask" or "delete subtask 'old task'"
    - Edit subtask details: "update first subtask details to 'contact via phone'"
    - Edit subtask datetime: "change second subtask deadline to tomorrow"
    
    Request body:
    {
        "prompt": "change task name to 'Complete report' and add subtask 'gather data'",
        "project_id": "abc123"
    }
    
    Returns: Updated task object
    """
    project_id = request.project_id
    prompt = request.prompt.strip()
    
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
    
    # Fetch the task
    task, found_project_id = await _find_task_across_projects(task_id, project_id)
    
    if not task:
        raise HTTPException(status_code=404, detail=f"Task with ID '{task_id}' not found")
    
    logging.info(f"=== EDIT TASK START ===")
    logging.info(f"Task ID: {task_id}, Project ID: {project_id}")
    logging.info(f"Original task: {json.dumps(task, indent=2)}")
    logging.info(f"Prompt: {prompt}")
    
    actual_project_id = project_id if project_id else found_project_id
    
    if not actual_project_id:
        raise HTTPException(status_code=400, detail="Could not determine project_id for this task")
    
    # Parse and apply edits
    try:
        edited_task = await _parse_and_apply_task_edits(task, prompt)
        logging.info(f"Edited task: {json.dumps(edited_task, indent=2)}")
    except Exception as e:
        logging.error(f"Error parsing prompt: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to parse edit instructions: {str(e)}")
    
    # Ensure all datetimes are valid and after now
    edited_task["datetime"] = _ensure_datetime_after_now_iso(edited_task.get("datetime", ""))

    for subtask in edited_task.get("subtasks", []):
        if isinstance(subtask, dict):
            subtask["datetime"] = _ensure_datetime_after_now_iso(subtask.get("datetime", ""))
    
    # Ensure subtask datetimes are before task datetime
    _ensure_subtasks_before_task_datetimes([edited_task])
    
    logging.info(f"Final task after datetime adjustments: {json.dumps(edited_task, indent=2)}")
    
    # Persist changes
    success = await _update_task_in_project(actual_project_id, task_id, edited_task)
    
    logging.info(f"Persistence success: {success}")
    logging.info(f"=== EDIT TASK END ===")
    
    if not success:
        logging.warning(f"Failed to persist task changes to project {actual_project_id}")
        # Prepare response data
        response_task = _prepare_task_response(edited_task)
        
        return {
            "success": False,
            "statusCode": 500,
            "message": "Task updated locally but failed to persist to backend",
            "data": response_task
        }
    
    # Prepare clean response data
    response_task = _prepare_task_response(edited_task)
    
    return {
        "success": True,
        "statusCode": 200,
        "message": "Task successfully updated",
        "data": response_task
    }


def _prepare_task_response(task: dict) -> dict:
    """
    Prepare task data for response by cleaning up internal fields and ensuring correct structure.
    Removes extra 'datetime' fields from subtasks while keeping original fields.
    """
    response_task = {
        "task": task.get("task", ""),
        "details": task.get("details", ""),
        "taskDueDate": task.get("datetime", task.get("taskDueDate", "")),
        "isDeleted": task.get("isDeleted", False),
        "isComplite": task.get("isComplite", False),
        "isStar": task.get("isStar", False),
        "_id": task.get("_id", "")
    }
    
    # Clean up subtasks - remove the extra 'datetime' field we added
    subtasks = []
    for subtask in task.get("subtasks", []):
        if isinstance(subtask, dict):
            clean_subtask = {
                "title": subtask.get("subtask", subtask.get("title", "")),
                "subTaskDueDate": subtask.get("subTaskDueDate", subtask.get("datetime", "")),
                "isStar": subtask.get("isStar", False),
                "isDeleted": subtask.get("isDeleted", False),
                "isComplite": subtask.get("isComplite", False),
                "_id": subtask.get("_id", "")
            }
            subtasks.append(clean_subtask)
    
    response_task["subtasks"] = subtasks
    
    return response_task



async def _find_task_across_projects(task_id: str, hint_project_id: Optional[str] = None) -> Tuple[Optional[dict], Optional[str]]:
    """
    Find a task by task_id, optionally using hint_project_id as a starting point.
    Returns (task_dict, project_id) or (None, None) if not found.
    """
    # First try with hint_project_id if provided
    if hint_project_id:
        try:
            task, success = await _fetch_task_by_id(hint_project_id, task_id)
            if success and task:
                return task, hint_project_id
        except Exception as e:
            logging.warning(f"Failed to fetch task from hint project {hint_project_id}: {e}")
    
    # Try in-memory projects
    for proj_id, project in projects.items():
        tasks = project.get("tasks", [])
        
        # Search by index
        if task_id.isdigit():
            idx = int(task_id)
            if 0 <= idx < len(tasks):
                task = tasks[idx]
                if isinstance(task, dict):
                    return task, str(proj_id)
                return {"task": str(task), "details": "", "datetime": "", "subtasks": []}, str(proj_id)
        
        # Search by ID field
        for task in tasks:
            if isinstance(task, dict):
                if task.get("id") == task_id or task.get("_id") == task_id:
                    return task, str(proj_id)
    
    return None, None

async def _parse_and_apply_task_edits(task: dict, prompt: str) -> dict:
    """
    Parse natural language prompt and apply edits to the task.
    Returns the edited task dictionary.
    """
    prompt_lower = prompt.lower().strip()
    
    # Log the original task for debugging
    logging.info(f"Original task datetime: {task.get('datetime')}")
    logging.info(f"Edit prompt: {prompt}")
    
    # Use LLM to parse the edit instructions
    parse_prompt = f"""
You are a task editing assistant. Given a task and edit instructions, return a JSON object with the edits to apply.

Current task:
{json.dumps(task, indent=2)}

User edit instructions: "{prompt}"

CRITICAL: When the user says "extend the datetime X days from now" or similar, they mean to ADD X days to the CURRENT task datetime.

Return a JSON object with these optional fields (only include fields that should be changed):
- task_name: new task name (string)
- details: new task details (string)  
- datetime_instruction: natural language datetime instruction
- add_subtasks: array of subtasks to add, each with {{subtask, details, datetime_instruction}}
- edit_subtasks: array of edits, each with {{index_or_name, field, new_value}} where field is "subtask", "details", or "datetime_instruction"
- remove_subtasks: array of subtask indices or names to remove
- replace_subtasks: array with {{index_or_name, subtask, details, datetime_instruction}}

For datetime_instruction, use these formats:
- If extending/postponing from current: "extend 3 days" or "add 5 days"
- If setting absolute date: "2025-11-15" or "tomorrow" or "5 days from now"

Examples:

Input: "extend the datetime 3 days from now"
Output: {{"datetime_instruction": "extend 3 days"}}

Input: "extend deadline by 5 days"  
Output: {{"datetime_instruction": "extend 5 days"}}

Input: "postpone by 1 week"
Output: {{"datetime_instruction": "extend 7 days"}}

Input: "set deadline to 2025-11-15"
Output: {{"datetime_instruction": "2025-11-15"}}

Return ONLY the JSON object, no other text.
"""
    
    try:
        response = await generate_text_with_context(parse_prompt)
        cleaned = re.sub(r'```(?:json)?\s*', '', response, flags=re.IGNORECASE).replace('```', '').strip()
        
        # Extract JSON
        match = re.search(r'\{.*\}', cleaned, flags=re.DOTALL)
        if match:
            edits = json.loads(match.group(0))
            logging.info(f"LLM parsed edits: {edits}")
        else:
            edits = {}
            logging.warning("No JSON found in LLM response")
    except Exception as e:
        logging.error(f"LLM parsing failed: {e}")
        # Fallback to simple regex parsing
        edits = _fallback_parse_edit_prompt(prompt, task)
        logging.info(f"Fallback parsed edits: {edits}")
    
    # Apply edits to task
    edited_task = task.copy()
    edited_task.setdefault("subtasks", [])
    
    # Edit task name
    if edits.get("task_name"):
        edited_task["task"] = edits["task_name"]
    
    # Edit task details
    if edits.get("details"):
        edited_task["details"] = edits["details"]
    
    # Edit task datetime - CRITICAL FIX
    if edits.get("datetime_instruction"):
        instruction = edits["datetime_instruction"].lower().strip()
        logging.info(f"Processing datetime instruction: {instruction}")
        
        # Get current task datetime
        current_dt_str = edited_task.get("datetime", "")
        current_dt = _parse_iso_like(current_dt_str)
        
        logging.info(f"Current datetime parsed: {current_dt}")
        
        # Check if this is an extension/addition
        if any(keyword in instruction for keyword in ["extend", "add", "postpone", "delay", "push"]):
            if current_dt:
                new_dt = _parse_datetime_instruction_from_base(instruction, current_dt)
                logging.info(f"Extended datetime to: {new_dt}")
            else:
                # If no valid current datetime, use now as base
                new_dt = _parse_datetime_instruction_from_base(instruction, _now_utc())
                logging.info(f"No current datetime, using now as base: {new_dt}")
            edited_task["datetime"] = new_dt
        else:
            # Absolute datetime setting
            new_dt = _parse_datetime_instruction(instruction)
            logging.info(f"Set datetime to: {new_dt}")
            edited_task["datetime"] = new_dt
    
    # Add subtasks
    if edits.get("add_subtasks"):
        for sub_data in edits["add_subtasks"]:
            new_subtask = {
                "subtask": sub_data.get("subtask", "New subtask"),
                "details": sub_data.get("details", ""),
                "datetime": _parse_datetime_instruction(sub_data.get("datetime_instruction", "3 days from now"))
            }
            edited_task["subtasks"].append(new_subtask)
    
    # Edit subtasks
    if edits.get("edit_subtasks"):
        for edit in edits["edit_subtasks"]:
            idx_or_name = edit.get("index_or_name")
            field = edit.get("field")
            new_value = edit.get("new_value")
            
            # Find subtask
            subtask_idx = _find_subtask_index(edited_task["subtasks"], idx_or_name)
            if subtask_idx is not None:
                if field == "subtask":
                    edited_task["subtasks"][subtask_idx]["subtask"] = new_value
                elif field == "details":
                    edited_task["subtasks"][subtask_idx]["details"] = new_value
                elif field == "datetime_instruction":
                    instruction = new_value.lower().strip()
                    if any(keyword in instruction for keyword in ["extend", "add", "postpone", "delay"]):
                        current_dt = _parse_iso_like(edited_task["subtasks"][subtask_idx].get("datetime", ""))
                        if current_dt:
                            edited_task["subtasks"][subtask_idx]["datetime"] = _parse_datetime_instruction_from_base(instruction, current_dt)
                        else:
                            edited_task["subtasks"][subtask_idx]["datetime"] = _parse_datetime_instruction_from_base(instruction, _now_utc())
                    else:
                        edited_task["subtasks"][subtask_idx]["datetime"] = _parse_datetime_instruction(instruction)
    
    # Replace subtasks
    if edits.get("replace_subtasks"):
        for replacement in edits["replace_subtasks"]:
            idx_or_name = replacement.get("index_or_name")
            subtask_idx = _find_subtask_index(edited_task["subtasks"], idx_or_name)
            if subtask_idx is not None:
                edited_task["subtasks"][subtask_idx] = {
                    "subtask": replacement.get("subtask", "Replaced subtask"),
                    "details": replacement.get("details", ""),
                    "datetime": _parse_datetime_instruction(replacement.get("datetime_instruction", "3 days from now"))
                }
    
    # Remove subtasks
    if edits.get("remove_subtasks"):
        indices_to_remove = []
        for idx_or_name in edits["remove_subtasks"]:
            subtask_idx = _find_subtask_index(edited_task["subtasks"], idx_or_name)
            if subtask_idx is not None:
                indices_to_remove.append(subtask_idx)
        
        # Remove in reverse order to maintain indices
        for idx in sorted(indices_to_remove, reverse=True):
            edited_task["subtasks"].pop(idx)
    
    logging.info(f"Final edited task datetime: {edited_task.get('datetime')}")
    return edited_task



def _parse_datetime_instruction_from_base(instruction: str, base_datetime: datetime) -> str:
    """
    Parse datetime instruction relative to a base datetime (for extensions).
    Examples: "extend 3 days", "extend 2 weeks", "add 5 hours"
    """
    instruction_lower = instruction.lower().strip()
    
    logging.info(f"Parsing datetime from base: {instruction_lower}, base: {base_datetime}")
    
    # Extract the duration - more flexible patterns
    patterns = [
        r'(\d+)\s+(day|hour|week|month)s?',  # "3 days", "5 hours"
        r'(?:extend|add|postpone|delay|push)\s+(?:by\s+)?(\d+)\s+(day|hour|week|month)s?',
    ]
    
    match = None
    for pattern in patterns:
        match = re.search(pattern, instruction_lower)
        if match:
            break
    
    if match:
        amount = int(match.group(1))
        unit = match.group(2)
        
        logging.info(f"Extracted: {amount} {unit}(s)")
        
        if unit == "hour":
            dt = base_datetime + timedelta(hours=amount)
        elif unit == "day":
            dt = base_datetime + timedelta(days=amount)
        elif unit == "week":
            dt = base_datetime + timedelta(weeks=amount)
        elif unit == "month":
            dt = base_datetime + timedelta(days=amount * 30)
        else:
            dt = base_datetime + timedelta(days=amount)
        
        result = _ensure_datetime_after_now_iso(dt.isoformat())
        logging.info(f"Calculated new datetime: {result}")
        return result
    
    # Fallback: add 1 day
    logging.warning("Could not parse duration, defaulting to +1 day")
    dt = base_datetime + timedelta(days=1)
    return _ensure_datetime_after_now_iso(dt.isoformat())



def _fallback_parse_edit_prompt(prompt: str, task: dict) -> dict:
    """
    Fallback regex-based parsing when LLM fails.
    Enhanced to catch more datetime extension patterns.
    """
    prompt_lower = prompt.lower()
    edits = {}
    
    # Parse task name change
    name_match = re.search(r'(?:change|update|rename|set)\s+(?:task\s+)?name\s+(?:to\s+)?["\']([^"\']+)["\']', prompt_lower)
    if name_match:
        edits["task_name"] = name_match.group(1).strip().title()
    
    # Parse details change
    details_match = re.search(r'(?:change|update|set)\s+details\s+(?:to\s+)?["\']([^"\']+)["\']', prompt_lower)
    if details_match:
        edits["details"] = details_match.group(1).strip()
    
    # Parse datetime change - MULTIPLE PATTERNS
    # Pattern 1: "extend datetime 3 days from now"
    extend_match = re.search(r'extend\s+(?:the\s+)?(?:datetime|deadline|date)\s+(\d+)\s+(day|hour|week|month)s?', prompt_lower)
    if extend_match:
        amount = extend_match.group(1)
        unit = extend_match.group(2)
        edits["datetime_instruction"] = f"extend {amount} {unit}s"
        logging.info(f"Fallback caught: extend {amount} {unit}s")
    else:
        # Pattern 2: "extend by 3 days" or "postpone by 3 days"
        extend_match2 = re.search(r'(?:extend|postpone|delay|push\s+back)\s+(?:by\s+)?(\d+)\s+(day|hour|week|month)s?', prompt_lower)
        if extend_match2:
            amount = extend_match2.group(1)
            unit = extend_match2.group(2)
            edits["datetime_instruction"] = f"extend {amount} {unit}s"
            logging.info(f"Fallback caught: extend {amount} {unit}s")
        else:
            # Pattern 3: Regular datetime change
            datetime_match = re.search(r'(?:change|update|set)\s+(?:deadline|date|datetime)\s+(?:to\s+)?(.+?)(?:\s+and|\.|$)', prompt_lower)
            if datetime_match:
                edits["datetime_instruction"] = datetime_match.group(1).strip()
    
    # Parse add subtask
    add_match = re.search(r'add\s+subtask\s+["\']([^"\']+)["\']', prompt_lower)
    if add_match:
        edits["add_subtasks"] = [{
            "subtask": add_match.group(1).strip().title(),
            "details": f"Complete: {add_match.group(1).strip()}",
            "datetime_instruction": "3 days from now"
        }]
    
    # Parse remove subtask
    if "remove" in prompt_lower or "delete" in prompt_lower:
        if "last" in prompt_lower:
            edits["remove_subtasks"] = [-1]
        elif "first" in prompt_lower:
            edits["remove_subtasks"] = [0]
    
    return edits


def _find_subtask_index(subtasks: List[dict], idx_or_name) -> Optional[int]:
    """
    Find subtask index by numeric index or by name match.
    Supports negative indices (e.g., -1 for last subtask).
    """
    if isinstance(idx_or_name, int):
        if idx_or_name < 0:
            # Negative index
            actual_idx = len(subtasks) + idx_or_name
            if 0 <= actual_idx < len(subtasks):
                return actual_idx
        elif 0 <= idx_or_name < len(subtasks):
            return idx_or_name
    elif isinstance(idx_or_name, str):
        # Try as number first
        if idx_or_name.lstrip('-').isdigit():
            return _find_subtask_index(subtasks, int(idx_or_name))
        
        # Try to find by name match
        idx_or_name_lower = idx_or_name.lower()
        for i, sub in enumerate(subtasks):
            if isinstance(sub, dict):
                sub_name = sub.get("subtask", "").lower()
                if idx_or_name_lower in sub_name or sub_name in idx_or_name_lower:
                    return i
    
    return None


def _parse_datetime_instruction(instruction: str) -> str:
    """
    Parse natural language datetime instruction into ISO datetime string.
    Examples: "3 days from now", "tomorrow", "2025-11-15", "next week"
    """
    if not instruction:
        return _random_future_datetime_iso()
    
    instruction_lower = instruction.lower().strip()
    now = _now_utc()
    
    # Try parsing as ISO date first
    parsed = _parse_iso_like(instruction)
    if parsed:
        return _ensure_datetime_after_now_iso(parsed.isoformat())
    
    # Handle relative dates
    if "tomorrow" in instruction_lower:
        dt = now + timedelta(days=1)
        return dt.replace(hour=9, minute=0, second=0, microsecond=0).isoformat() + "Z"
    
    if "today" in instruction_lower:
        dt = now + timedelta(hours=2)  # 2 hours from now
        return dt.replace(minute=0, second=0, microsecond=0).isoformat() + "Z"
    
    if "next week" in instruction_lower:
        dt = now + timedelta(days=7)
        return dt.replace(hour=9, minute=0, second=0, microsecond=0).isoformat() + "Z"
    
    # Extract "X days/hours from now"
    match = re.search(r'(\d+)\s+(day|hour|week|month)s?\s+(?:from\s+now)?', instruction_lower)
    if match:
        amount = int(match.group(1))
        unit = match.group(2)
        
        if unit == "hour":
            dt = now + timedelta(hours=amount)
        elif unit == "day":
            dt = now + timedelta(days=amount)
        elif unit == "week":
            dt = now + timedelta(weeks=amount)
        elif unit == "month":
            dt = now + timedelta(days=amount * 30)
        else:
            dt = now + timedelta(days=1)
        
        return dt.replace(minute=0, second=0, microsecond=0).isoformat() + "Z"
    
    # Fallback
    return _random_future_datetime_iso()


def _fallback_parse_edit_prompt(prompt: str, task: dict) -> dict:
    """
    Fallback regex-based parsing when LLM fails.
    """
    prompt_lower = prompt.lower()
    edits = {}
    
    # Parse task name change
    name_match = re.search(r'(?:change|update|rename|set)\s+(?:task\s+)?name\s+(?:to\s+)?["\']([^"\']+)["\']', prompt_lower)
    if name_match:
        edits["task_name"] = name_match.group(1).strip().title()
    
    # Parse details change
    details_match = re.search(r'(?:change|update|set)\s+details\s+(?:to\s+)?["\']([^"\']+)["\']', prompt_lower)
    if details_match:
        edits["details"] = details_match.group(1).strip()
    
    # Parse datetime change
    datetime_match = re.search(r'(?:change|update|set)\s+(?:deadline|date|datetime)\s+(?:to\s+)?(.+?)(?:\s+and|\.|$)', prompt_lower)
    if datetime_match:
        edits["datetime_instruction"] = datetime_match.group(1).strip()
    
    # Parse add subtask
    add_match = re.search(r'add\s+subtask\s+["\']([^"\']+)["\']', prompt_lower)
    if add_match:
        edits["add_subtasks"] = [{
            "subtask": add_match.group(1).strip().title(),
            "details": f"Complete: {add_match.group(1).strip()}",
            "datetime_instruction": "3 days from now"
        }]
    
    # Parse remove subtask
    if "remove" in prompt_lower or "delete" in prompt_lower:
        if "last" in prompt_lower:
            edits["remove_subtasks"] = [-1]
        elif "first" in prompt_lower:
            edits["remove_subtasks"] = [0]
    
    return edits


async def _update_task_in_project(project_id: str, task_id: str, updated_task: dict) -> bool:
    """
    Update a specific task in a project and persist to backend.
    Returns True if successful, False otherwise.
    """
    # Fetch the full project
    try:
        project, external_fetched, numeric_idx = await _fetch_project_by_id(project_id)
        
        if not project:
            return False
        
        tasks = project.get("tasks", [])
        task_idx = None
        
        # Find the task to update
        if task_id.isdigit():
            idx = int(task_id)
            if 0 <= idx < len(tasks):
                task_idx = idx
        
        if task_idx is None:
            # Search by ID
            for i, task in enumerate(tasks):
                if isinstance(task, dict):
                    if task.get("id") == task_id or task.get("_id") == task_id:
                        task_idx = i
                        break
        
        if task_idx is None:
            logging.error(f"Task {task_id} not found in project {project_id}")
            return False
        
        # Update the task
        tasks[task_idx] = updated_task
        
        # Persist to in-memory if applicable
        if numeric_idx is not None:
            projects[numeric_idx]["tasks"] = tasks
        
        # Persist to external service
        if external_fetched:
            base = os.getenv("PROJECT_SERVICE_URL")
            if not base:
                return False
            
            update_path = f"{base.rstrip('/')}/api/v1/project/update/{project_id}/"
            payload = {"tasks": tasks}
            
            async with httpx.AsyncClient() as client:
                try:
                    resp = await client.patch(update_path, json=payload, timeout=10.0)
                    if resp.status_code in (200, 201, 204):
                        return True
                    logging.warning(f"Failed to update project: {resp.status_code} {resp.text}")
                except Exception as e:
                    logging.error(f"Exception updating project: {e}")
                    return False
        
        return True
        
    except Exception as e:
        logging.error(f"Error updating task in project: {e}")
        return False