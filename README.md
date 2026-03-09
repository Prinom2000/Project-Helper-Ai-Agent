# Project Task AI Agent

A FastAPI-based AI agent system designed to assist with project task management through intelligent conversation and automated task creation. The system leverages Large Language Models (LLMs) to understand user queries, provide project insights, and generate tasks based on project context.

## Features

- **Intelligent Task Creation**: Automatically generates tasks from user conversations using LLM-powered intent detection
- **Project Question Answering**: Answer questions about project details, features, and requirements
- **Conversation Context**: Maintains context from previous interactions to improve task generation
- **LLM Integration**: Supports OpenAI and DeepSeek models via OpenRouter
- **RESTful API**: Clean FastAPI endpoints for easy integration
- **Docker Support**: Containerized deployment with Docker Compose
- **Database Integration**: Seamless integration with project management databases

## Prerequisites

- Python 3.11+
- Docker (optional, for containerized deployment)
- OpenRouter API key (for LLM services)
- Project service URL (for database operations)

## Installation

### Local Development

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd "Project Task Ai Agent"
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   Create a `.env` file in the root directory:
   ```env
   OPENAI_API_KEY=your_openai_api_key
   OPENROUTER_API_KEY=your_openrouter_api_key
   PROJECT_SERVICE_URL=your_project_service_base_url
   ```

### Docker Deployment

1. **Build and run with Docker Compose**:
   ```bash
   docker-compose up --build
   ```

The application will be available at `http://localhost:8000`.

## Configuration

The application uses the following environment variables:

- `OPENAI_API_KEY`: API key for OpenAI services
- `OPENROUTER_API_KEY`: API key for OpenRouter (required for DeepSeek LLM)
- `PROJECT_SERVICE_URL`: Base URL for the project service API

## Running the Application

### Local Development

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Docker

```bash
docker-compose up
```

Visit `http://localhost:8000/docs` for the interactive API documentation.

## API Endpoints

### Dynamic Agent Endpoints

#### POST `/dynamic_agent/task_ask/`
Ask questions about a project and store the conversation context.

**Request Body**:
```json
{
  "userId": "string",
  "projectId": "string",
  "text": "string"
}
```

**Response**: Project-related answer based on the query.

#### POST `/dynamic_agent/task_create/`
Create a new task based on user query and conversation context.

**Request Body**:
```json
{
  "userId": "string",
  "projectId": "string",
  "query": "string"
}
```

**Response**:
```json
{
  "success": true,
  "intent": "create_task",
  "reason": "string",
  "task": {
    "title": "string",
    "description": "string",
    "compliteTarget": "string"
  },
  "database_response": {}
}
```

### Other Endpoints

- `/projects/*`: Project and task management endpoints
- `/`: Health check endpoint

## Testing

Run the test suite:

```bash
python -m pytest test/
```

Or run quick tests:

```bash
python test/run_quick_tests.py
```

## Project Structure

```
Project Task Ai Agent/
├── app/
│   ├── api/
│   │   └── v1/
│   │       └── endpoints/
│   │           ├── agent.py
│   │           ├── dynamic_agent.py
│   │           ├── project_task_question.py
│   │           └── ProjectUpdateAgent.py
│   ├── schemas/
│   │   └── project.py
│   ├── services/
│   │   └── openai_service.py
│   ├── utils/
│   │   └── task_utils.py
│   ├── config.py
│   └── main.py
├── config/
│   └── config.yaml
├── test/
│   ├── project.json
│   ├── run_quick_tests.py
│   ├── t2.py
│   └── test_project_update_agent.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── IMPLEMENTATION_SUMMARY.md
└── README.md
```

## Deployment

### Docker Compose

The application includes a `docker-compose.yml` for easy deployment:

```bash
docker-compose up -d
```

### Production Considerations

- Set appropriate environment variables
- Configure reverse proxy (nginx, etc.)
- Set up monitoring and logging
- Use production-grade database connections

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions or issues, please open an issue on the GitHub repository.
