from typing import Dict, List, Optional, Union

import pydantic


class Metric(pydantic.BaseModel):
    name: str
    type: Optional[str]
    groups: List[Dict[str, int]]
    step_axis: Optional[str]
    data: Union[Dict, List]
