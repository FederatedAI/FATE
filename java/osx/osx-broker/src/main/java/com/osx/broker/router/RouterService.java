package com.osx.broker.router;


import com.osx.api.router.RouterInfo;

public interface RouterService {
    RouterInfo route(String srcPartyId, String srcRole, String dstPartyId, String desRole);
}
