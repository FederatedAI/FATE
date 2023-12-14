from typing import Optional

from fate.arch.unify import uuid


def generate_computing_uuid(session_id: Optional[str] = None):
    if session_id is None:
        return f"computing_{uuid()}"
    else:
        return f"{session_id}_computing_{uuid()}"
