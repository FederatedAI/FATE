package org.fedai.osx.broker.router;


import org.fedai.osx.api.router.RouterInfo;

public interface RouterService {
    RouterInfo route(String srcPartyId, String srcRole, String dstPartyId, String desRole);
}
