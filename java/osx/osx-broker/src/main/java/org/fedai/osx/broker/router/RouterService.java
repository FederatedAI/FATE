package org.fedai.osx.broker.router;


import org.fedai.osx.core.router.RouterInfo;

public interface RouterService {
    RouterInfo route(String srcPartyId, String srcRole, String dstPartyId, String desRole);

    RouterInfo routePtp(String srcInstId,String srcNodeId,String dstInstId,String dstNodeId);
}
