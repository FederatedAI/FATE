from fate_arch.common import Party


def _parse_runtime_conf(runtime_conf):
    role = runtime_conf.get("local").get("role")
    party_id = str(runtime_conf.get("local").get("party_id"))
    party = Party(role, party_id)
    parties = {}
    for role, pid_list in runtime_conf.get("role", {}).items():
        parties[role] = [Party(role, pid) for pid in pid_list]
    return party, parties
