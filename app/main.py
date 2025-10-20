# main.py

from fastapi import FastAPI
from app.api.v1.endpoints import project_task_question  # Import the consolidated file

app = FastAPI()

# Include the router for project, task, and question endpoints
app.include_router(project_task_question.router, prefix="/projects", tags=["projects"])

@app.get("/")
async def read_root():
    return {"message": "Welcome to Go Get A Genie! Start your project."}
