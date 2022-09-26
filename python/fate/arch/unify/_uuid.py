from typing import Optional
from uuid import uuid1


def uuid():
    return uuid1().hex


def generate_computing_uuid(session_id: Optional[str] = None):
    if session_id is None:
        return f"computing_{uuid()}"
    else:
        return f"{session_id}_computing_{uuid()}"
