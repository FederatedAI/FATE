from typing import Literal

T_GUEST = Literal["guest"]
T_HOST = Literal["host"]
T_ARBITER = Literal["arbiter"]
T_ROLE = Literal[T_GUEST, T_HOST, T_ARBITER]
