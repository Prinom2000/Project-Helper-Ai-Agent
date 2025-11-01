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
    - Edit task datetime: "as user promt"
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

Current date/time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

User edit instructions: "{prompt}"

Return a JSON object with these optional fields (only include fields that should be changed):
- task_name: new task name (string)
- details: new task details (string)
- details_action: "append", "replace", "remove", or "enhance" - specifies how to handle details
- datetime_instruction: natural language datetime instruction
- priority: task priority (string: "low", "medium", "high")
- status: task status (string: "pending", "in_progress", "completed")
- add_subtasks: array of subtasks to add, each with {{subtask, details, datetime_instruction}}
- edit_subtasks: array of edits, each with {{index_or_name, field, new_value}} where field is "subtask", "details", or "datetime_instruction"
- remove_subtasks: array of subtask indices or names to remove
- replace_subtasks: array with {{index_or_name, subtask, details, datetime_instruction}}

DETAILS EDITING RULES:
1. ADD/APPEND new content to existing details:
   - details_action: "append"
   - details: "the new content to add"
   - Result: existing details + new content
  
2. REPLACE all details with new content:
   - details_action: "replace"
   - details: "the new complete details"
   - Result: completely new details (old details discarded)
  
3. ENHANCE/MAKE MORE DESCRIPTIVE:
   - details_action: "enhance"
   - details: "enhanced version of the existing details"
   - Use this when user asks to make details more descriptive, elaborate, detailed, or comprehensive
   - Take the existing details and expand them with more context, specifics, and clarity
   - Result: improved version of existing details
  
4. REMOVE all details:
   - details_action: "remove"
   - details: "" (can be empty)
   - Result: no details

DATETIME INSTRUCTION RULES:
1. EXTENDING/POSTPONING from current task datetime:
   - "extend 3 days" - add 3 days to current task datetime
   - "add 5 days" - add 5 days to current task datetime
   - "postpone 2 weeks" - add 14 days to current task datetime
   - "move 48 hours" - add 48 hours to current task datetime

2. ABSOLUTE DATE/TIME settings:
   - ISO format: "2025-11-15", "2025-12-03T21:30:00"
   - Natural dates with year: "January 31, 2026", "November 5, 2025", "December 15, 2026"
   - Abbreviated dates with year: "jan 31, 2026", "nov 5, 2025", "dec 15, 2026"
   - Day-month-year format: "25 jan, 2036", "15 dec, 2025", "5 nov, 2026"
   - Casual/Short formats: "11 dec", "5 nov", "dec 11", "15th", "tomorrow"
   - Numeric date formats: "26-6-2026", "6-26-2026", "26/6/2026", "6/26/2026", "26.6.2026"
   - Relative to NOW: "tomorrow", "in 3 days", "next Friday at 6 PM", "two weeks from now"
   - Specific day: "Monday, 4th November, 2:00 PM", "next Monday"
   - Week/Month references: "first week of December", "last Friday of November", "first Monday of December 2025"
   - Same day patterns: "same day next month", "same weekday next month"
   - Month name variants: Accept both abbreviated and full month names
   - Date number only: "15" or "15th" (use current/next month based on context)

3. RELATIVE TO ANOTHER TASK/EVENT:
   - "2 days before due date"
   - "15 minutes before event starts"
   - "one day before main task"
   - "two days after approval"
   - "48 hours after current deadline"

4. CONDITIONAL/BUSINESS LOGIC:
   - "next working day after December 1st"
   - "if deadline falls on weekend, move to next Monday"
   - "three business days from today"
   - "second Tuesday of December"

5. RECURRING/REPEATING:
   - "repeat weekly"
   - "every Sunday at midnight"
   - "every 2nd and 4th Thursday"
   - "every 3 days until done"

MONTH ABBREVIATION MAPPING:
Always convert abbreviated months to full month names in the output:
- jan → January
- feb → February
- mar → March
- apr → April
- may → May
- jun → June
- jul → July
- aug → August
- sep → September
- oct → October
- nov → November
- dec → December

NUMERIC DATE PARSING:
Handle various numeric date formats and convert to ISO or readable format:
- "DD-MM-YYYY" format: "26-6-2026" → "2026-06-26"
- "MM-DD-YYYY" format: "6-26-2026" → "2026-06-26"
- "DD/MM/YYYY" format: "26/6/2026" → "2026-06-26"
- "MM/DD/YYYY" format: "6/26/2026" → "2026-06-26"
- "DD.MM.YYYY" format: "26.6.2026" → "2026-06-26"
- Ambiguous dates: If day ≤ 12, assume DD-MM-YYYY format by default
- If day > 12, it's clearly the day (e.g., "26-6-2026" means day=26, month=6)
- Single digit months/days are acceptable: "5-6-2026", "26-6-2026"

DATE WITH YEAR PARSING:
Handle dates that include year in various formats:
- "Month DD, YYYY": "January 31, 2026", "December 15, 2025" → "2026-01-31", "2025-12-15"
- "mon DD, YYYY": "jan 31, 2026", "dec 15, 2025" → "2026-01-31", "2025-12-15"
- "DD mon, YYYY": "25 jan, 2036", "15 dec, 2025" → "2036-01-25", "2025-12-15"
- "DD Month, YYYY": "25 January, 2036", "15 December, 2025" → "2036-01-25", "2025-12-15"
- Accept with or without comma: "jan 31 2026", "25 jan 2036", "January 31 2026"
- Always convert to ISO format (YYYY-MM-DD) when year is provided

DATETIME PARSING INTELLIGENCE:
- Parse casual date formats and convert to full month names: "11 dec" → "December 11", "5 nov" → "November 5"
- Accept abbreviated months (jan, feb, mar, apr, may, jun, jul, aug, sep, oct, nov, dec) and convert to full names
- Handle day-month and month-day orders: "11 dec" and "dec 11" both become "December 11"
- Parse dates with year: "January 31, 2026" → "2026-01-31", "jan 31, 2026" → "2026-01-31", "25 jan, 2036" → "2036-01-25"
- Parse numeric dates: "26-6-2026", "6/26/2026", "26.6.2026" all convert to "2026-06-26"
- If only day number given (like "15th"), infer current or next month
- Accept dates without year (default to current year or next year if past)
- Normalize time formats: "10am", "10 am", "10:00 AM" all work
- Handle informal language: "deadline 11 dec", "due dec 11", "by 11th december"
- Always output ISO format (YYYY-MM-DD) when year is provided in input
- Output readable format when year is NOT provided: "December 11" (not "dec 11")
- FLEXIBLE DATE KEYWORD RECOGNITION: Recognize any variation of "deadline", "due date", "deu date", "in date", "date will be", "date =", "due =", "deadline =", "due date =", "set date", "change date", "schedule date", "by date", "on date", "for date", "at date" — all treated as setting the task's datetime_instruction
- DATE FORMAT FLEXIBILITY: Accept "december 5", "5 december", "5 dec", "dec 5", "december 5th", "5th december", "5th dec", "dec 5th", "december 5 2025", "5 december 2025", "dec 5 2025", "January 31, 2026", "jan 31, 2026", "25 jan, 2036", "26-6-2026", "6/26/2026", etc.

SUBTASK DATETIME DEPENDENCIES:
- For subtasks with relative dates to parent: "one day before main task", "two days after parent completion", "same day as parent"
- Use "depends_on_parent" prefix: "depends_on_parent: -1 day", "depends_on_parent: +2 days"

EXAMPLES:

Input: "deadline January 31, 2026"
Output: {{"datetime_instruction": "2026-01-31"}}

Input: "set date to jan 31, 2026"
Output: {{"datetime_instruction": "2026-01-31"}}

Input: "due date 25 jan, 2036"
Output: {{"datetime_instruction": "2036-01-25"}}

Input: "deadline 15 December, 2025"
Output: {{"datetime_instruction": "2025-12-15"}}

Input: "set to December 5, 2026"
Output: {{"datetime_instruction": "2026-12-05"}}

Input: "due nov 20, 2025"
Output: {{"datetime_instruction": "2025-11-20"}}

Input: "schedule 10 mar 2026"
Output: {{"datetime_instruction": "2026-03-10"}}

Input: "deadline February 28 2026"
Output: {{"datetime_instruction": "2026-02-28"}}

Input: "set the deadline 26-6-2026"
Output: {{"datetime_instruction": "2026-06-26"}}

Input: "deadline = 15-12-2025"
Output: {{"datetime_instruction": "2025-12-15"}}

Input: "due date 5/11/2025"
Output: {{"datetime_instruction": "2025-11-05"}}

Input: "set date to 20.3.2026"
Output: {{"datetime_instruction": "2026-03-20"}}

Input: "deadline 1-1-2026"
Output: {{"datetime_instruction": "2026-01-01"}}

Input: "due 31/12/2025"
Output: {{"datetime_instruction": "2025-12-31"}}

Input: "schedule for 10-5-2026"
Output: {{"datetime_instruction": "2026-05-10"}}

Input: "deadline = december 5"
Output: {{"datetime_instruction": "December 5"}}

Input: "due date=5 dec"
Output: {{"datetime_instruction": "December 5"}}

Input: "deu date = dec 5"
Output: {{"datetime_instruction": "December 5"}}

Input: "in date = 5 december"
Output: {{"datetime_instruction": "December 5"}}

Input: "date will be=december 5"
Output: {{"datetime_instruction": "December 5"}}

Input: "date = 5 dec"
Output: {{"datetime_instruction": "December 5"}}

Input: "deadline=5th december"
Output: {{"datetime_instruction": "December 5th"}}

Input: "due=dec 15"
Output: {{"datetime_instruction": "December 15"}}

Input: "set date to 15 jan"
Output: {{"datetime_instruction": "January 15"}}

Input: "change due date to 20 feb"
Output: {{"datetime_instruction": "February 20"}}

Input: "schedule for 10 mar"
Output: {{"datetime_instruction": "March 10"}}

Input: "by date = 5 apr"
Output: {{"datetime_instruction": "April 5"}}

Input: "on 12 aug"
Output: {{"datetime_instruction": "August 12"}}

Input: "for 8 sep"
Output: {{"datetime_instruction": "September 8"}}

Input: "at oct 22"
Output: {{"datetime_instruction": "October 22"}}

Input: "deadline = 5 december at 3 PM"
Output: {{"datetime_instruction": "December 5 at 3 PM"}}

Input: "due date=dec 11, 10:30 AM"
Output: {{"datetime_instruction": "December 11 at 10:30 AM"}}

Input: "date will be = 15th nov"
Output: {{"datetime_instruction": "November 15th"}}

Input: "extend the datetime 3 days from now"
Output: {{"datetime_instruction": "extend 3 days"}}

Input: "extend deadline by 5 days"
Output: {{"datetime_instruction": "extend 5 days"}}

Input: "postpone by 1 week"
Output: {{"datetime_instruction": "extend 7 days"}}

Input: "move deadline 48 hours after current deadline"
Output: {{"datetime_instruction": "extend 48 hours"}}

Input: "set deadline to 2025-11-15"
Output: {{"datetime_instruction": "2025-11-15"}}

Input: "set deadline to 11 dec"
Output: {{"datetime_instruction": "December 11"}}

Input: "set date: 2025-12-11"
Output: {{"datetime_instruction": "2025-12-11"}}

Input: "deadline 15 nov"
Output: {{"datetime_instruction": "November 15"}}

Input: "due date dec 20"
Output: {{"datetime_instruction": "December 20"}}

Input: "change to 5th"
Output: {{"datetime_instruction": "5th"}}

Input: "deadline 20 jan"
Output: {{"datetime_instruction": "January 20"}}

Input: "set to 15 feb"
Output: {{"datetime_instruction": "February 15"}}

Input: "due mar 10"
Output: {{"datetime_instruction": "March 10"}}

Input: "schedule 5 apr"
Output: {{"datetime_instruction": "April 5"}}

Input: "deadline 30 may"
Output: {{"datetime_instruction": "May 30"}}

Input: "set to 18 jun"
Output: {{"datetime_instruction": "June 18"}}

Input: "due jul 25"
Output: {{"datetime_instruction": "July 25"}}

Input: "deadline 12 aug"
Output: {{"datetime_instruction": "August 12"}}

Input: "set to 8 sep"
Output: {{"datetime_instruction": "September 8"}}

Input: "due oct 22"
Output: {{"datetime_instruction": "October 22"}}

Input: "Set the meeting date to 5th November at 10 AM"
Output: {{"datetime_instruction": "November 5th at 10 AM"}}

Input: "Schedule for next Friday at 6 PM"
Output: {{"datetime_instruction": "next Friday at 6 PM"}}

Input: "Change to last Friday of November"
Output: {{"datetime_instruction": "last Friday of November"}}

Input: "Task due date will be November 20th"
Output: {{"datetime_instruction": "November 20th"}}

Input: "Deadline is in 3 days"
Output: {{"datetime_instruction": "in 3 days"}}

Input: "Due date two weeks from now"
Output: {{"datetime_instruction": "two weeks from now"}}

Input: "set to 11 december"
Output: {{"datetime_instruction": "December 11"}}

Input: "deadline by dec 15th"
Output: {{"datetime_instruction": "December 15"}}

Input: "reschedule to 20 nov"
Output: {{"datetime_instruction": "November 20"}}

Input: "Add subtask 'Buy helmet', due tomorrow"
Output: {{"add_subtasks": [{{"subtask": "Buy helmet", "details": "", "datetime_instruction": "tomorrow"}}]}}

Input: "Add subtask 'Check tire', due in 3 days"
Output: {{"add_subtasks": [{{"subtask": "Check tire pressure", "details": "", "datetime_instruction": "in 3 days"}}]}}

Input: "Create subtask 'Find documents', date next Monday"
Output: {{"add_subtasks": [{{"subtask": "Find old documents", "details": "", "datetime_instruction": "next Monday"}}]}}

Input: "Add subtask 'Email supplier', deadline 2025-11-06"
Output: {{"add_subtasks": [{{"subtask": "Email supplier", "details": "", "datetime_instruction": "2025-11-06"}}]}}

Input: "Add subtask 'Email supplier', deadline 6 nov"
Output: {{"add_subtasks": [{{"subtask": "Email supplier", "details": "", "datetime_instruction": "November 6"}}]}}

Input: "Add subtask 'Email supplier', deadline 15-11-2025"
Output: {{"add_subtasks": [{{"subtask": "Email supplier", "details": "", "datetime_instruction": "2025-11-15"}}]}}

Input: "Add subtask 'Email supplier', deadline jan 15, 2026"
Output: {{"add_subtasks": [{{"subtask": "Email supplier", "details": "", "datetime_instruction": "2026-01-15"}}]}}

Input: "Add subtask 'Call client', deadline 15 jan"
Output: {{"add_subtasks": [{{"subtask": "Call client", "details": "", "datetime_instruction": "January 15"}}]}}

Input: "Create subtask 'Submit report', due 10 mar"
Output: {{"add_subtasks": [{{"subtask": "Submit report", "details": "", "datetime_instruction": "March 10"}}]}}

Input: "Add subtask 'Review files', date 22 aug"
Output: {{"add_subtasks": [{{"subtask": "Review files", "details": "", "datetime_instruction": "August 22"}}]}}

Input: "Add subtask 'Review files', date 22/8/2026"
Output: {{"add_subtasks": [{{"subtask": "Review files", "details": "", "datetime_instruction": "2026-08-22"}}]}}

Input: "Add subtask 'Meeting', date December 20, 2026"
Output: {{"add_subtasks": [{{"subtask": "Meeting", "details": "", "datetime_instruction": "2026-12-20"}}]}}

Input: "Add subtask 'Verify supplier', must be done one day before main task"
Output: {{"add_subtasks": [{{"subtask": "Verify supplier details", "details": "", "datetime_instruction": "depends_on_parent: -1 day"}}]}}

Input: "Create subtask 'Collect documents', scheduled two days after approval"
Output: {{"add_subtasks": [{{"subtask": "Collect documents", "details": "", "datetime_instruction": "depends_on_parent: +2 days"}}]}}

Input: "add details: call my mom"
Output: {{"details": "call my mom", "details_action": "append"}}

Input: "Add more details to the task, mention the time and reason"
Output: {{"details": "Time and reason need to be specified", "details_action": "append"}}

Input: "Update task with items and estimated cost"
Output: {{"details": "Items list and estimated cost", "details_action": "append"}}

Input: "set details: call my mom"
Output: {{"details": "call my mom", "details_action": "replace"}}

Input: "Update the task details with new information about the meeting"
Output: {{"details": "New information about the meeting", "details_action": "replace"}}

Input: "Change details to: Buy groceries including milk, eggs, bread"
Output: {{"details": "Buy groceries including milk, eggs, bread", "details_action": "replace"}}

Input: "add a more descriptive details"
Current details: "Research different bike options, visit local bike shops, test ride bikes, compare prices, purchase chosen bike"
Output: {{"details": "Research various types of bikes online, shortlist potential models based on budget and features, visit multiple local bike shops for expert advice, take several bikes for test rides to compare comfort and performance, evaluate price differences and available discounts, and finally purchase the bike that best meets your needs and preferences", "details_action": "enhance"}}

Input: "make the task description more detailed"
Current details: "Call mom"
Output: {{"details": "Call mom to check on her health, ask about her day, discuss upcoming family events, and ensure she has everything she needs", "details_action": "enhance"}}

Input: "elaborate on the task details"
Current details: "Buy groceries"
Output: {{"details": "Buy groceries from the local supermarket, including fresh vegetables, fruits, dairy products like milk and eggs, bread, and essential pantry items. Check for weekly discounts and use the shopping list to avoid missing items", "details_action": "enhance"}}

Input: "make it more comprehensive"
Current details: "Prepare presentation"
Output: {{"details": "Prepare a comprehensive presentation by researching the topic thoroughly, creating an outline with key points, designing engaging slides with visuals and data, practicing the delivery multiple times, and preparing for potential questions from the audience", "details_action": "enhance"}}

Input: "add more descriptive content"
Current details: "Fix bug"
Output: {{"details": "Fix the bug by first reproducing the issue in the development environment, analyzing error logs and stack traces, identifying the root cause in the codebase, implementing a proper fix, writing test cases to prevent regression, and thoroughly testing the solution before deployment", "details_action": "enhance"}}

Input: "Remove the description, keep only title"
Output: {{"details_action": "remove"}}

Input: "Delete all additional notes"
Output: {{"details_action": "remove"}}

Input: "Clear task details"
Output: {{"details_action": "remove"}}

Input: "Add details and mark as high priority"
Output: {{"details": "Additional details", "details_action": "append", "priority": "high"}}

Input: "Set reminder for 15 minutes before event starts"
Output: {{"datetime_instruction": "15 minutes before event"}}

Input: "Reschedule to 48 hours after current deadline"
Output: {{"datetime_instruction": "extend 48 hours"}}

Input: "Move due date two weeks earlier"
Output: {{"datetime_instruction": "extend -14 days"}}

Input: "Shift deadline to same day next month"
Output: {{"datetime_instruction": "same day next month"}}

Input: "Schedule for first Monday of December 2025"
Output: {{"datetime_instruction": "first Monday of December 2025"}}

Input: "Set to next working day after December 1st"
Output: {{"datetime_instruction": "next working day after December 1st"}}

Input: "Schedule for three business days from today"
Output: {{"datetime_instruction": "three business days from today"}}

CRITICAL RULES:
- ALWAYS distinguish between "extend/add/postpone X days" (relative to current task datetime) vs "in X days" (relative to now)
- If user says "extend", "add more", "postpone", "delay" - use "extend X days/hours"
- If user says "set to", "change to", "schedule for", "due date is", "deadline", "deadline =", "due =", "due date =", "deu date =", "in date =", "date will be", "date =", "by date", "on date", "for date", "at date" - use absolute datetime instruction
- For subtask dependencies on parent task, use "depends_on_parent: +/-X days/hours"
- For details: use details_action to specify "append", "replace", "remove", or "enhance"
- When adding/appending details, provide the content to be added WITHOUT prefixes
- When replacing details, provide the complete new content WITHOUT prefixes
- When enhancing details, EXPAND the existing details to make them more descriptive, comprehensive, and detailed
- When removing details, set details_action to "remove"
- ENHANCE action keywords: "more descriptive", "more detailed", "elaborate", "comprehensive", "enrich", "expand on", "add more context"
- ALWAYS convert month abbreviations to full names: jan→January, feb→February, mar→March, apr→April, may→May, jun→June, jul→July, aug→August, sep→September, oct→October, nov→November, dec→December
- Parse dates with year in ISO format: "January 31, 2026" → "2026-01-31", "jan 31, 2026" → "2026-01-31", "25 jan, 2036" → "2036-01-25"
- Accept formats: "Month DD, YYYY", "mon DD, YYYY", "DD mon, YYYY", "DD Month, YYYY" (with or without commas)
- Parse numeric date formats: "26-6-2026" → "2026-06-26", "15/12/2025" → "2025-12-15", "10.3.2026" → "2026-03-10"
- For numeric dates with separators (-./), assume DD-MM-YYYY format unless day > 12, then it's obvious
- Accept single or double digit days/months: "5-6-2026" and "05-06-2026" are both valid
- Parse casual date formats intelligently: "11 dec" → "December 11", "5 jan" → "January 5", "mar 20" → "March 20"
- When year is provided, ALWAYS output in ISO format (YYYY-MM-DD)
- When year is NOT provided, output in readable format: "December 11" (not "dec 11")
- For numeric dates, output in ISO format: "26-6-2026" → "2026-06-26"
- Parse time components carefully: "10 AM", "21:30", "08:45", "11:59 PM", "10am", "10 am"
- Handle various date formats: ISO, natural language with year, abbreviated with year, casual shorthand, numeric formats (DD-MM-YYYY, DD/MM/YYYY, DD.MM.YYYY), relative expressions, and flexible keyword prefixes
- Be flexible with date input variations: "January 31, 2026", "jan 31, 2026", "25 jan, 2036", "11 dec", "dec 11", "26-6-2026", "26/6/2026", etc.

Return ONLY the JSON object, no other text or explanation.
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