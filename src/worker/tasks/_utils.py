"""Shared helpers for worker tasks."""


def _serialize_event(event: dict) -> dict:
    """Convert a LangGraph streaming event dict to a JSON-safe dict."""
    safe: dict = {}
    for key, value in event.items():
        if key == "messages":
            safe["messages"] = [
                {"type": msg.type, "content": msg.content}
                for msg in value
            ]
        elif isinstance(value, (str, int, float, bool, list, dict)) or value is None:
            safe[key] = value
        else:
            safe[key] = str(value)
    return safe
