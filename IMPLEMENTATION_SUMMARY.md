# Implementation Summary: Task Ask & Task Create Routes

## Overview
Implemented functionality to store user messages from the `task_ask` route and use them to create tasks via a new `task_create` route.

---

## Changes Made

### 1. Global Variable for Message Storage
**Location:** [dynamic_agent.py](app/api/v1/endpoints/dynamic_agent.py#L20-L25)

Added a global variable `last_messege` to store the last user message from the `task_ask` route:
```python
last_messege = {
    "userId": None,
    "projectId": None,
    "text": None,
}
```

### 2. Updated `task_ask` Route
**Location:** [dynamic_agent.py](app/api/v1/endpoints/dynamic_agent.py#L805-L851)

Modified the existing route to:
- Store user message in the `last_messege` global variable every time it's hit
- Keep all existing functionality for answering project questions
- Update the variable on each request

```python
@router.post("/task_ask/")
async def task_ask_route(payload: TaskPayload):
    global last_messege
    
    # Store last message in global variable
    last_messege = {
        "userId": user_id,
        "projectId": project_id,
        "text": user_text,
    }
    # ... rest of existing functionality
```

### 3. New Pydantic Model
**Location:** [dynamic_agent.py](app/api/v1/endpoints/dynamic_agent.py#L677-L681)

Added `TaskCreatePayload` model for the new route:
```python
class TaskCreatePayload(BaseModel):
    userId: str
    projectId: str
    query: str
```

### 4. Helper Functions for Task Creation
**Location:** [dynamic_agent.py](app/api/v1/endpoints/dynamic_agent.py#L854-L967)

#### a. `_detect_task_intent_with_deepseek()`
Analyzes user query using DeepSeek LLM to determine intent:
- **Intent:** `create_task` or `general_chat`
- **Uses:** DeepSeek model for intent detection
- **Returns:** JSON with intent and reason

#### b. `_generate_task_from_query()`
Generates task object using DeepSeek LLM based on:
- Project data context
- User query
- Last message from `task_ask` route
- **Returns:** Task object with `title`, `description`, and `compliteTarget` fields

#### c. `_push_task_to_database()`
Posts task to database endpoint:
- **Endpoint:** `POST {{baseUrl}}/api/v1/updateProject/:projectId/tasks`
- **Payload:** Contains userId, title, description, and compliteTarget
- **Returns:** Database response

### 5. New `task_create` Route
**Location:** [dynamic_agent.py](app/api/v1/endpoints/dynamic_agent.py#L970-L1026)

**Endpoint:** `POST /dynamic_agent/task_create/`

**Workflow:**
1. Accepts `userId`, `projectId`, and `query` from request
2. Fetches project data from: `GET {{baseUrl}}/api/v1/updateProject/getProject/:projectId`
3. Retrieves conversation from `last_messege` variable (from `task_ask` route)
4. Detects user intent using DeepSeek LLM:
   - If intent is `create_task`: Use the user query as task description
   - If intent is `general_chat`: Use the `last_messege` from `task_ask` route instead
5. Generates task using LLM with full project context
6. Pushes task to database
7. Returns success response with intent, reason, generated task, and database response

**Request Payload:**
```json
{
  "userId": "6970460d402853f36465a1d8",
  "projectId": "60b8c8a9c1e4a8b2c9d3e4f5",
  "query": "make a new task for this"
}
```

**Response Payload:**
```json
{
  "success": true,
  "intent": "create_task",
  "reason": "User explicitly asked to create a new task",
  "task": {
    "title": "New Task need money",
    "description": "This is a detailed description of the task.",
    "compliteTarget": "2024-12-31T23:59:59Z"
  },
  "database_response": {...}
}
```

---

## Usage Flow

### Step 1: Ask a Question
```
POST /dynamic_agent/task_ask/
{
  "userId": "6970460d402853f36465a1d8",
  "projectId": "60b8c8a9c1e4a8b2c9d3e4f5",
  "text": "What features should we include in the payment system?"
}
```
✅ Message stored in `last_messege` variable

### Step 2: Create Task
```
POST /dynamic_agent/task_create/
{
  "userId": "6970460d402853f36465a1d8",
  "projectId": "60b8c8a9c1e4a8b2c9d3e4f5",
  "query": "make a task for this"
}
```
✅ Intent detected, task created using `last_messege` context

---

## Key Features

✅ **Conversation Context:** Uses last message from `task_ask` route for better task generation  
✅ **Smart Intent Detection:** Determines if user wants to create task or just chatting  
✅ **LLM-Powered:** Uses DeepSeek LLM for intent detection and task generation  
✅ **Fallback Logic:** If no task creation intent, uses `last_messege` for task generation  
✅ **Full Project Context:** Task generation considers entire project data  
✅ **Database Integration:** Automatically pushes tasks to project database  

---

## Error Handling

The implementation includes proper error handling for:
- Missing environment variables
- Missing required payload fields
- Failed project data fetch
- Failed intent detection
- Failed task generation
- Failed database push

All errors return appropriate HTTP status codes with descriptive messages.

---

## Configuration Required

Ensure the following environment variables are set:
- `OPENROUTER_API_KEY` - For DeepSeek LLM calls
- `PROJECT_SERVICE_URL` - Base URL for project service endpoints
