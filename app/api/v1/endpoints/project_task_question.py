# api/v1/endpoints/project_task_question.py
import openai
import asyncio
from fastapi import APIRouter, HTTPException
from app.schemas.project import Answer
from app.services.openai_service import generate_text
from app.utils.task_utils import extract_tasks
import os
import httpx
from dotenv import load_dotenv
load_dotenv()
from typing import Tuple, Optional
import logging
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
@router.post("/answer_question/{project_id}/")
async def answer_question(project_id: str, answer: Answer):
    project, external_fetched, numeric_idx = await _fetch_project_by_id(project_id)

    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    if not project.get("answered_questions"):
        raise HTTPException(status_code=400, detail="No question to answer")

    # store answer in the most recent question
    project["answered_questions"][-1]["answer"] = answer.answer

    # persist locally if in-memory
    if numeric_idx is not None:
        projects[numeric_idx]["answered_questions"] = project["answered_questions"]

    # persist to external service when applicable
    if external_fetched:
        success = await _persist_answered_questions_to_service(project_id, project["answered_questions"])
        if not success:
            raise HTTPException(status_code=500, detail="Failed to persist answer to project service")

    # return full project details (normalized)
    return {"success": True, "project": project}

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
