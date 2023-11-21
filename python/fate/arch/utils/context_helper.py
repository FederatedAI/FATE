import os

from .paths import get_base_dir


def init_standalone_context(csession_id, federation_session_id, party, parties, data_dir=None):
    from fate.arch.computing.standalone import CSession
    from fate.arch.federation.standalone import StandaloneFederation
    from fate.arch.context import Context

    if not data_dir:
        data_dir = os.path.join(get_base_dir(), "data")

    computing_session = CSession(session_id=csession_id, data_dir=data_dir)
    federation_session = StandaloneFederation(computing_session, federation_session_id, party, parties)
    context = Context(computing=computing_session, federation=federation_session)
    return context
