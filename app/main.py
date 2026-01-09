# main.py

from fastapi import FastAPI
from app.api.v1.endpoints import project_task_question, agent, dynamic_agent, ProjectUpdateAgent  # Import the consolidated files

app = FastAPI()

# Include the routers for project, task, question and agent endpoints
app.include_router(project_task_question.router, prefix="/projects", tags=["projects"])
app.include_router(agent.router, prefix="/projects", tags=["projects"])
app.include_router(dynamic_agent.router, prefix="/dynamic_agent", tags=["dynamic_agent"])
app.include_router(ProjectUpdateAgent.router, prefix="/dynamic_agent", tags=["dynamic_agent"])

@app.get("/")
async def read_root():
    return {"message": "Welcome to Go Get A Genie! Start your project."}
