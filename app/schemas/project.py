# schemas/project.py

from pydantic import BaseModel

class UserInput(BaseModel):
    user_message: str

class ProjectGoal(BaseModel):
    add_task: str  # Task paragraph input

class Answer(BaseModel):
    answer: str  # User's answer to a generated question

class SubTask(BaseModel):
    subtask: str  # Subtask description

class TaskDetails(BaseModel):
    details: str  # Details for the task (optional)
