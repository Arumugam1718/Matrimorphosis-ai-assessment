"""Task 9: the notes and tasks MCP server. Run by docker compose / CI as:  python src/app/mcp_server/server.py --http  (port 9000, path /mcp)

The storage is PROVIDED (taskstore.py: TaskStore), the user lookup is PROVIDED (app/users.py: UserStore.user_for_token).
You write the integration: register the eight tools below on `mcp`, and make every call run AS THE CALLING USER.

How the server knows the user (task 9)
  The agent forwards the logged-in user's token on every MCP call as the HTTP header  Authorization: Bearer <token>  (you write that
  half in agent.py, task 9). A tool declares a `ctx: Context` parameter; FastMCP injects it and the HTTP request is
  ctx.request_context.request (Starlette: .headers). Write ONE helper `current_user_id(ctx) -> str` that
    - reads the bearer token, looks it up with UserStore().user_for_token(token)
    - raises ToolError("ERROR[UNAUTHENTICATED]: <what>. Next step: log in and retry.") when the header is missing, malformed or unknown
  and call it first in every tool. Never take a user id from a tool argument: the model must not be able to name another user.

The eight tools (names and arguments are fixed; each returns json.dumps(<what the TaskStore method returned>), a string)
  add_note(title: str, body: str)               list_notes()                          delete_note(note_id: int)
  add_task(title: str, details: str = "")       list_tasks(status: str = "")          set_task_status(task_id: int, status: str)
  delete_task(task_id: int)                     task_summary()
  status is "todo" | "in_progress" | "done". TaskStore raises TaskError for bad input: re-raise it as ToolError(str(error)).
  The docstring of each tool is its prompt for the model: say WHEN to use it and describe the arguments.
"""
import json
import os
import sys

from mcp.server.fastmcp import Context, FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from app.mcp_server.taskstore import TaskError, TaskStore
from app.users import UserStore

mcp = FastMCP("tasks", host="0.0.0.0", port=int(os.getenv("MCP_PORT", "9000")))
store = TaskStore()


def current_user_id(ctx: Context) -> str:
    """Extract and validate the bearer token from the HTTP request header, returning the user_id.
    Raises ToolError with ERROR[UNAUTHENTICATED] if the token is missing, malformed, or unknown.
    """
    try:
        auth = ctx.request_context.request.headers.get("authorization", "")
    except Exception:
        raise ToolError("ERROR[UNAUTHENTICATED]: No authorization header. Next step: log in and retry.")
    if not auth.startswith("Bearer ") and not auth.startswith("bearer "):
        raise ToolError("ERROR[UNAUTHENTICATED]: Missing or malformed Authorization header. Next step: log in and retry.")
    token = auth.split(" ", 1)[1].strip()
    if not token:
        raise ToolError("ERROR[UNAUTHENTICATED]: Empty token. Next step: log in and retry.")
    user = UserStore().user_for_token(token)
    if not user:
        raise ToolError("ERROR[UNAUTHENTICATED]: Unknown or expired token. Next step: log in and retry.")
    return user["user_id"]


@mcp.tool()
def add_note(title: str, body: str, ctx: Context) -> str:
    """Add a personal note with a title and body. Use this when the user wants to save a note,
    reminder, or piece of information for later retrieval.

    Args:
        title: short title for the note (required)
        body: the content of the note
    """
    user_id = current_user_id(ctx)
    try:
        result = store.add_note(user_id, title, body)
    except TaskError as exc:
        raise ToolError(str(exc)) from exc
    return json.dumps(result)


@mcp.tool()
def list_notes(ctx: Context) -> str:
    """List all notes saved by the current user. Use this when the user asks to see their notes,
    or when you need to find a specific note by its content.
    """
    user_id = current_user_id(ctx)
    try:
        result = store.list_notes(user_id)
    except TaskError as exc:
        raise ToolError(str(exc)) from exc
    return json.dumps(result)


@mcp.tool()
def delete_note(note_id: int, ctx: Context) -> str:
    """Delete a note by its id. Use this when the user wants to remove a note.
    Call list_notes first if you need to find the note_id.

    Args:
        note_id: the numeric id of the note to delete
    """
    user_id = current_user_id(ctx)
    try:
        result = store.delete_note(user_id, note_id)
    except TaskError as exc:
        raise ToolError(str(exc)) from exc
    return json.dumps(result)


@mcp.tool()
def add_task(title: str, ctx: Context, details: str = "") -> str:
    """Add a new task to the user's task list with status 'todo'. Use this when the user wants to
    track something they need to do.

    Args:
        title: short title for the task (required)
        details: optional longer description of the task
    """
    user_id = current_user_id(ctx)
    try:
        result = store.add_task(user_id, title, details)
    except TaskError as exc:
        raise ToolError(str(exc)) from exc
    return json.dumps(result)


@mcp.tool()
def list_tasks(ctx: Context, status: str = "") -> str:
    """List the user's tasks, optionally filtered by status. Use this when the user wants to see
    their tasks or check what is pending, in progress, or done.

    Args:
        status: filter by status: 'todo', 'in_progress', or 'done'. Leave empty to list all tasks.
    """
    user_id = current_user_id(ctx)
    try:
        result = store.list_tasks(user_id, status)
    except TaskError as exc:
        raise ToolError(str(exc)) from exc
    return json.dumps(result)


@mcp.tool()
def set_task_status(task_id: int, status: str, ctx: Context) -> str:
    """Update the status of a task. The allowed transitions are: todo -> in_progress,
    in_progress -> done, in_progress -> todo, done -> in_progress. Use this when the user
    reports progress on a task.

    Args:
        task_id: the numeric id of the task to update
        status: the new status: 'todo', 'in_progress', or 'done'
    """
    user_id = current_user_id(ctx)
    try:
        result = store.set_task_status(user_id, task_id, status)
    except TaskError as exc:
        raise ToolError(str(exc)) from exc
    return json.dumps(result)


@mcp.tool()
def delete_task(task_id: int, ctx: Context) -> str:
    """Delete a task by its id. Use this when the user wants to remove a task entirely.
    Call list_tasks first if you need to find the task_id.

    Args:
        task_id: the numeric id of the task to delete
    """
    user_id = current_user_id(ctx)
    try:
        result = store.delete_task(user_id, task_id)
    except TaskError as exc:
        raise ToolError(str(exc)) from exc
    return json.dumps(result)


@mcp.tool()
def task_summary(ctx: Context) -> str:
    """Return a summary of the user's tasks grouped by status (todo, in_progress, done).
    Use this when the user asks for an overview of their tasks or progress.
    """
    user_id = current_user_id(ctx)
    try:
        result = store.summary(user_id)
    except TaskError as exc:
        raise ToolError(str(exc)) from exc
    return json.dumps(result)


if __name__ == "__main__":
    # stdio cannot carry the Authorization header: the agent talks to this server over HTTP (MCP_URL)
    mcp.run(transport="streamable-http" if "--http" in sys.argv else "stdio")
