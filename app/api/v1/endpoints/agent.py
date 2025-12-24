from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import os
import httpx
import logging
from typing import Optional, List, Tuple

router = APIRouter()

logging.basicConfig(level=logging.INFO)


class LLMAgentRequest(BaseModel):
    userId: str = Field(..., alias="userId")
    query: str = Field(..., alias="query")  # any project/task/subtask name or partial text
    project_name: Optional[str] = Field(None, alias="project_name")

    class Config:
        allow_population_by_field_name = True
        extra = "ignore"


async def _fetch_projects_for_user(user_id: str) -> list:
    base = os.getenv("PROJECT_SERVICE_URL")
    if not base:
        logging.warning("PROJECT_SERVICE_URL not set in environment (.env)")
        return []

    path = f"{base.rstrip('/')}/api/v1/project/user/project/{user_id}/"
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(path, timeout=10.0)
        if resp.status_code != 200:
            logging.warning(f"Failed to fetch user projects: {resp.status_code} {resp.text}")
            return []
        payload = resp.json()
        projects_list = payload.get("data") if isinstance(payload, dict) and payload.get("data") else payload
        if isinstance(projects_list, dict):
            return [projects_list]
        if isinstance(projects_list, list):
            return projects_list
        return []
    except Exception as e:
        logging.warning(f"Exception fetching user projects: {e}")
        return []


def _match_project(projects: list, name: str) -> Tuple[Optional[dict], Optional[List[dict]]]:
    """Try to find a single best match. Return (single_match, multiple_matches_or_none)."""
    if not projects or not name:
        return None, None
    name_norm = name.strip().lower()

    # Exact match on common fields
    for p in projects:
        for field in ("goal", "project_goal", "title", "name"):
            try:
                v = p.get(field)
            except Exception:
                v = None
            if v and isinstance(v, str) and v.strip().lower() == name_norm:
                return p, None

    # Substring matches
    matches = []
    for p in projects:
        for field in ("goal", "project_goal", "title", "name"):
            try:
                v = p.get(field)
            except Exception:
                v = None
            if v and isinstance(v, str) and name_norm in v.strip().lower():
                matches.append(p)
                break

    if len(matches) == 1:
        return matches[0], None
    if len(matches) > 1:
        return None, matches

    return None, None


def _search_projects_by_query(projects: list, query: str) -> Tuple[Optional[dict], Optional[List[dict]]]:
    """Search projects, tasks and subtasks for `query`.

    Prioritize most specific exact match (subtask > task > project). If no exact match,
    collect substring matches across projects/tasks/subtasks. Returns (single_match, multiple_matches).
    single_match is a dict with a 'type' key and relevant objects.
    """
    if not projects or not query:
        return None, None
    q = query.strip().lower()

    # Exact matches: check project, then task, then subtask
    for p in projects:
        for field in ("goal", "project_goal", "title", "name"):
            v = p.get(field)
            if isinstance(v, str) and v.strip().lower() == q:
                return ({"type": "project", "project": p}, None)
        for t in p.get("tasks", []) or []:
            t_title = t.get("task") if isinstance(t.get("task"), str) else None
            if t_title and t_title.strip().lower() == q:
                return ({"type": "task", "project": p, "task": t}, None)
            for s in t.get("subtasks", []) or []:
                s_title = s.get("title") if isinstance(s.get("title"), str) else None
                if s_title and s_title.strip().lower() == q:
                    return ({"type": "subtask", "project": p, "task": t, "subtask": s}, None)

    # Substring matches
    matches = []
    for p in projects:
        # project substrings
        for field in ("goal", "project_goal", "title", "name"):
            v = p.get(field)
            if isinstance(v, str) and q in v.strip().lower():
                matches.append({"type": "project", "project": p})
                break
        for t in p.get("tasks", []) or []:
            t_title = t.get("task") if isinstance(t.get("task"), str) else ""
            if t_title and q in t_title.strip().lower():
                matches.append({"type": "task", "project": p, "task": t})
                continue
            for s in t.get("subtasks", []) or []:
                s_title = s.get("title") if isinstance(s.get("title"), str) else ""
                if s_title and q in s_title.strip().lower():
                    matches.append({"type": "subtask", "project": p, "task": t, "subtask": s})
                    continue

    if not matches:
        return None, None

    if len(matches) == 1:
        return (matches[0], None)

    # Build minimal descriptors for multiple matches
    minimal = []
    for m in matches:
        typ = m.get("type")
        p = m.get("project")
        proj_id = str((p.get("_id") or p.get("id") or "")) if p else ""
        item = {"type": typ, "project_id": proj_id, "project_goal": p.get("goal") or p.get("title") or p.get("name") or ""}
        if typ == "task":
            t = m.get("task")
            item["task_id"] = str((t.get("_id") or t.get("id") or "")) if t else ""
            item["task_title"] = t.get("task") if t else ""
        if typ == "subtask":
            t = m.get("task")
            s = m.get("subtask")
            item["task_id"] = str((t.get("_id") or t.get("id") or "")) if t else ""
            item["subtask_id"] = str((s.get("_id") or s.get("id") or "")) if s else ""
            item["subtask_title"] = s.get("title") if s else ""
        minimal.append(item)

    return None, minimal


@router.post("/LLM_Agent/")
async def LLM_Agent(payload: LLMAgentRequest):
    """Search for a project, task, or subtask for a user's query.

    Request JSON examples:
      { "userId": "...", "query": "Research game-based learning benefits" }
      (backwards compatible) { "userId": "...", "project_name": "Game Learning Initiative" }

    Responses:
      - Single match: returns `match_type` and the `project` plus `task`/`subtask` when applicable.
      - Multiple matches: returns `matches` (minimal descriptors) and `count` for disambiguation.
    """
    user_id = payload.userId
    query = (payload.query or payload.project_name or "").strip()

    if not user_id or not query:
        raise HTTPException(status_code=400, detail="Both userId and query (or project_name) are required")

    projects_list = await _fetch_projects_for_user(user_id)
    if not projects_list:
        raise HTTPException(status_code=404, detail="No projects found for this user")

    single, multiple = _search_projects_by_query(projects_list, query)

    if single:
        result = {"match_type": single.get("type")}
        if single.get("type") == "project":
            p = single.get("project")
            result["project"] = p
            result["project_id"] = str(p.get("_id") or p.get("id") or "")
        elif single.get("type") == "task":
            p = single.get("project")
            t = single.get("task")
            result["project"] = p
            result["project_id"] = str(p.get("_id") or p.get("id") or "")
            result["task"] = t
            result["task_id"] = str(t.get("_id") or t.get("id") or "")
        elif single.get("type") == "subtask":
            p = single.get("project")
            t = single.get("task")
            s = single.get("subtask")
            result["project"] = p
            result["project_id"] = str(p.get("_id") or p.get("id") or "")
            result["task"] = t
            result["task_id"] = str(t.get("_id") or t.get("id") or "")
            result["subtask"] = s
            result["subtask_id"] = str(s.get("_id") or s.get("id") or "")
        return result

    if multiple:
        return {"matches": multiple, "count": len(multiple)}

    raise HTTPException(status_code=404, detail="No project/task/subtask matching the given query was found for this user")
