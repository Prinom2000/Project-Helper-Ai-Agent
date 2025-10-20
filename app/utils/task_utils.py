# utils/task_utils.py

import re
from fastapi import HTTPException

async def extract_tasks(task_paragraph: str):
    try:
        # Use regex to split tasks based on common delimiters (e.g., period, exclamation mark, newline)
        task_list = re.split(r'(?<=\.)\s*(?=[A-Z])', task_paragraph)
        
        # Clean up each task to remove any unwanted extra spaces or special characters
        tasks = [task.strip() for task in task_list if task.strip()]
        
        return tasks
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting tasks: {str(e)}")
