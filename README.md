# Project Task Management API

## Overview
A comprehensive FastAPI-based backend service for intelligent project and task management with AI-powered features. The system provides natural language processing capabilities for task generation, editing, and project assistance.
## LIVE👉 https://ai.gogetagenie.com/docs

## Features

### Core Functionality
- **Project Management**: Create, retrieve, and update projects with goals and Q&A
- **AI-Powered Task Generation**: Automatically generate tasks and subtasks from project goals
- **Natural Language Task Editing**: Edit tasks using conversational prompts
- **Intelligent Chat Assistant**: OLLIE - project-specific assistant for queries
- **Smart Title Generation**: Auto-generate concise project titles
- **DateTime Intelligence**: Advanced datetime parsing and scheduling

### Advanced Capabilities
- **Natural Language Prompts**: Create and modify tasks using everyday language
- **Subtask Management**: Automatic subtask generation with dependencies
- **DateTime Flexibility**: Support for absolute, relative, and dependency-based scheduling
- **Details Management**: Append, replace, enhance, or remove task descriptions
- **Multi-Format Support**: ISO dates, natural language, numeric formats (DD-MM-YYYY, DD/MM/YYYY)

## Tech Stack
- **Framework**: FastAPI
- **AI/ML**: OpenAI GPT-3.5-turbo
- **HTTP Client**: httpx (async)
- **Environment**: Python 3.8+
- **Dependencies**: pydantic, python-dotenv, asyncio

## Installation

### Prerequisites
```bash
Python 3.8+
pip package manager
```

### Setup
```bash
# Clone repository
git clone <repository-url>
cd project-task-api

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install fastapi uvicorn openai httpx python-dotenv pydantic

# Create .env file
cp .env.example .env
```

### Environment Variables
Create a `.env` file in the project root:
```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# External Project Service URL (optional)
PROJECT_SERVICE_URL=http://your-project-service.com
```

## API Endpoints

### 1. Ask Question
Generate intelligent questions about project goals.

**Endpoint**: `POST /ask/{project_id}/`

**Example**:
```bash
curl -X POST "http://localhost:8000/ask/abc123/"
```

**Response**:
```json
{
  "question": "What is the target timeline for project completion?"
}
```

### 2. Chat with Assistant
Interact with OLLIE, the project assistant.

**Endpoint**: `POST /chat/{project_id}/`

**Parameters**:
- `user_message` (string): Your question or request

**Example**:
```bash
curl -X POST "http://localhost:8000/chat/abc123/" \
  -H "Content-Type: application/json" \
  -d '{"user_message": "What tasks are pending?"}'
```

**Response**:
```json
{
  "response": "You have 3 pending tasks: Design mockups, Code backend API, and Write documentation."
}
```

### 3. Get Project Details
Retrieve complete project information.

**Endpoint**: `GET /get_project/{project_id}/`

**Example**:
```bash
curl "http://localhost:8000/get_project/abc123/"
```

### 4. Generate Project Title
Create concise project titles from descriptions.

**Endpoint**: `POST /generate_title/`

**Request Body**:
```json
{
  "user_text ": "I want to build a mobile app for tracking daily water intake and reminding users to stay hydrated"
}
```

**Response**:
```json
{
  "title": "Daily Water Tracking App"
}
```

### 5. Get Project Tasks
Retrieve or generate tasks for a project.

**Endpoint**: `GET /project_tasks/{project_id}/`

**Query Parameters**:
- `prompt` (optional): Natural language instructions for task creation

**Examples**:

**Basic retrieval**:
```bash
curl "http://localhost:8000/project_tasks/abc123/"
```

**Create task via prompt**:
```bash
curl "http://localhost:8000/project_tasks/abc123/?prompt=make%20a%20task%20named%20call%20mom%20and%20deadline%20will%20be%20after%203%20days"
```

**Response**:
```json
{
  "project_id": "abc123",
  "tasks": [
    {
      "task": "Call mom",
      "details": "Contact mother",
      "datetime": "2025-11-16T09:00:00Z",
      "subtasks": [
        {
          "subtask": "Prepare talking points",
          "details": "List topics to discuss",
          "datetime": "2025-11-15T14:00:00Z"
        }
      ]
    }
  ]
}
```

### 6. Get Single Task
Retrieve details of a specific task.

**Endpoint**: `GET /task/{project_id}/{task_id}`

**Example**:
```bash
curl "http://localhost:8000/task/abc123/task_001"
```

**Response**:
```json
{
  "success": true,
  "task": {
    "task": "Design UI mockups",
    "details": "Create wireframes and high-fidelity designs",
    "datetime": "2025-11-20T09:00:00Z",
    "subtasks": [...]
  },
  "project_id": "abc123"
}
```

### 7. Edit Task (Natural Language)
Modify tasks using conversational prompts.

**Endpoint**: `PATCH /task/{task_id}/edit`

**Request Body**:
```json
{
  "prompt": "change task name to 'Complete quarterly report' and extend deadline 5 days",
  "project_id": "abc123"
}
```

**Supported Edit Operations**:

#### Task Name
```json
{"prompt": "change task name to 'Buy groceries'"}
{"prompt": "rename to 'Call supplier'"}
```

#### Task Details
```json
{"prompt": "add details: call my mom about birthday"}
{"prompt": "set details: comprehensive project review"}
{"prompt": "make details more descriptive"}
{"prompt": "remove all details"}
```

#### DateTime Operations
```json
// Extend from current deadline
{"prompt": "extend deadline 3 days"}
{"prompt": "postpone by 2 weeks"}

// Absolute dates
{"prompt": "set deadline to January 31, 2026"}
{"prompt": "deadline = 15-12-2025"}
{"prompt": "due date dec 11"}

// Relative to now
{"prompt": "deadline tomorrow"}
{"prompt": "due in 5 days"}
```

#### Subtask Management
```json
// Add
{"prompt": "add subtask 'Email client' due tomorrow"}

// Edit
{"prompt": "change first subtask name to 'Call supplier'"}
{"prompt": "update second subtask deadline to next Monday"}

// Remove
{"prompt": "remove last subtask"}
{"prompt": "delete subtask 'old task'"}
```

**Response**:
```json
{
  "success": true,
  "statusCode": 200,
  "message": "Task successfully updated",
  "data": {
    "task": "Complete quarterly report",
    "details": "...",
    "taskDueDate": "2025-11-18T09:00:00Z",
    "subtasks": [...]
  }
}
```

## DateTime Format Support

### Supported Formats
```javascript
// ISO Standard
"2025-11-15"
"2025-12-03T21:30:00Z"

// Natural Language with Year
"January 31, 2026"
"jan 31, 2026"
"25 jan, 2036"

// Casual/Short
"11 dec"
"dec 15"
"tomorrow"
"next Friday at 6 PM"

// Numeric Formats
"26-6-2026"  // DD-MM-YYYY
"6/26/2026"  // MM/DD/YYYY
"26.6.2026"  // DD.MM.YYYY

// Relative
"in 3 days"
"two weeks from now"
"extend 5 days"

// Dependencies
"one day before main task"
"48 hours after approval"
```

### Month Abbreviations
Automatically converts: jan→January, feb→February, mar→March, etc.

## Task Details Operations

### Append (Add to existing)
```json
{"prompt": "add details: call my mom"}
```
Result: Existing details + " call my mom"

### Replace (Complete overwrite)
```json
{"prompt": "set details: new complete description"}
```
Result: Only "new complete description"

### Enhance (Make more descriptive)
```json
{"prompt": "make details more comprehensive"}
```
Result: AI-enhanced, expanded version of existing details

### Remove (Clear all)
```json
{"prompt": "remove all details"}
```
Result: Empty details field

## Running the Server

### Development Mode
```bash
uvicorn main:app --reload --port 8000
```

### Production Mode
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Docker (Optional)
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Architecture


### Data Flow
```
User Request → FastAPI Endpoint → AI Processing (OpenAI) → 
Response Generation → Validation → External Service Sync → 
User Response
```

## Key Components

### 1. AI-Powered Question Generation
Generates contextual questions based on project goals and previous Q&A history.

### 2. Natural Language Task Parser
Converts human prompts into structured task operations:
- Extracts task names, details, deadlines
- Identifies edit operations (add, replace, remove, enhance)
- Parses complex datetime instructions

### 3. DateTime Intelligence Engine
Handles multiple datetime formats and operations:
- Absolute dates (ISO, natural language, numeric)
- Relative dates (from now, from task deadline)
- Dependency-based scheduling
- Business logic (working days, weekends)

### 4. Subtask Generator
Automatically creates 2-5 relevant subtasks per task with:
- Intelligent subtask naming
- Contextual descriptions
- Dependency-aware scheduling

### 5. Project Persistence Layer
Manages data across:
- In-memory storage (numeric IDs)
- External project service (string IDs)
- Multiple API endpoints and payload formats

## Error Handling

### Common Error Responses
```json
// 404 - Not Found
{
  "status_code": 404,
  "detail": "Project not found"
}

// 400 - Bad Request
{
  "status_code": 400,
  "detail": "No question to answer"
}

// 500 - Server Error
{
  "status_code": 500,
  "detail": "Error generating response: ..."
}
```

### Graceful Degradation
- Falls back to local storage if external service unavailable
- Uses regex parsing if AI parsing fails
- Provides default values for missing fields

## Best Practices

### 1. API Usage
```python
# Always provide project_id for edits
payload = {
    "prompt": "your edit instruction",
    "project_id": "abc123"  # Required
}

# Use specific, clear prompts
"change task name to 'Review contract' and deadline dec 15"
# Better than: "update task"
```

### 2. DateTime Specifications
```python
# Be explicit about extensions
"extend deadline 3 days"  # From current deadline
"deadline in 3 days"      # From now

# Use full dates with year for clarity
"January 31, 2026"  # Clear
"jan 31"            # Assumes current/next year
```

### 3. Details Editing
```python
# Specify action clearly
"add details: ..."      # Appends
"set details: ..."      # Replaces
"enhance details"       # AI expands
"remove details"        # Clears
```

## Logging

### Configuration
```python
logging.basicConfig(level=logging.INFO)
```

### Log Levels
- **INFO**: Normal operations, API calls, parsing results
- **WARNING**: Fallback usage, persistence failures
- **ERROR**: Exceptions, critical failures

### Key Log Messages
```
"=== EDIT TASK START ==="
"Original task datetime: ..."
"LLM parsed edits: ..."
"Persistence success: True/False"
"=== EDIT TASK END ==="
```

## Testing

### Sample Test Cases
```bash
# Test task creation
curl -X GET "http://localhost:8000/project_tasks/test123/?prompt=create%20task%20review%20documents%20deadline%205%20days"

# Test task editing
curl -X PATCH "http://localhost:8000/task/task_001/edit" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "extend deadline 3 days", "project_id": "test123"}'

# Test datetime parsing
curl -X PATCH "http://localhost:8000/task/task_001/edit" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "deadline january 31, 2026", "project_id": "test123"}'
```

## Limitations

### Current Constraints
- OpenAI API rate limits apply
- In-memory storage is not persistent across restarts
- External service dependency for production use
- No built-in authentication/authorization
- Single language support (English)

### Known Issues
- Complex multi-operation edits may require multiple API calls
- Ambiguous date formats may be interpreted differently
- Very long task lists may hit token limits

## Roadmap

### Planned Features
- [ ] Multi-language support
- [ ] Task priority management
- [ ] Recurring task scheduling
- [ ] Team collaboration features
- [ ] Advanced analytics and insights
- [ ] Webhook support for real-time updates
- [ ] GraphQL API option
- [ ] Enhanced security and authentication

## Requirements.txt

```txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
openai==0.28.1
httpx==0.25.1
python-dotenv==1.0.0
pydantic==2.5.0
```

## Deploy:
- **Digitalocean**


### Version 1.0.0 (Current)
- Initial release
- AI-powered task generation
- Natural language task editing
- Advanced datetime parsing
- Subtask management
- Project chat assistant

---

**Built with ❤️ using FastAPI and OpenAI**


