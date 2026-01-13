from app.api.v1.endpoints.ProjectUpdateAgent import _merge_ids_and_normalize_dates


def test_merge_ids_and_dates_edit():
    existing = {
        "_id": "695f767ae2fcb20149917442",
        "task": "update the old task",
        "taskDueDate": "2026-03-01T00:00:00Z",
        "subtasks": [
            {"_id": "695f767ae2fcb20149917444", "title": "update the old subtask", "subTaskDueDate": "2026-01-30T00:00:00Z"}
        ]
    }

    new_task = {
        "task": "update the old task",
        "details": "Set up the server and database.",
        "taskDueDate": "2026-03-01",
        "isDeleted": False,
        "isComplite": False,
        "isArchived": False,
        "isStar": False,
        "subtasks": [
            {"title": "update the old subtask", "subTaskDueDate": "2026-01-30"},
            {"title": "new subtask", "subTaskDueDate": "2026-02-01"}
        ]
    }

    merged = _merge_ids_and_normalize_dates('edit', existing, new_task)

    assert merged.get('_id') == "695f767ae2fcb20149917442"
    assert merged['subtasks'][0].get('_id') == "695f767ae2fcb20149917444"
    assert '_id' not in merged['subtasks'][1]
    assert merged['taskDueDate'] == "2026-03-01T00:00:00Z"
    assert merged['subtasks'][0]['subTaskDueDate'] == "2026-01-30T00:00:00Z"
    assert merged['subtasks'][1]['subTaskDueDate'] == "2026-02-01T00:00:00Z"


def test_merge_ids_and_dates_add():
    new_task = {
        "task": "add new task",
        "taskDueDate": "2026-03-01",
        "subtasks": [
            {"title": "add new subtask", "subTaskDueDate": "2026-01-30", "_id": "should_remove"}
        ]
    }

    merged = _merge_ids_and_normalize_dates('add', None, new_task)

    assert '_id' not in merged
    assert '_id' not in merged['subtasks'][0]
    assert merged['taskDueDate'] == "2026-03-01T00:00:00Z"
    assert merged['subtasks'][0]['subTaskDueDate'] == "2026-01-30T00:00:00Z"


if __name__ == '__main__':
    test_merge_ids_and_dates_edit()
    test_merge_ids_and_dates_add()
    print('All quick tests passed')