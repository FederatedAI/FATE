from typing import Optional

import pydantic


class ArtifactSpec(pydantic.BaseModel):
    name: str
    uri: str
    metadata: Optional[dict] = None
