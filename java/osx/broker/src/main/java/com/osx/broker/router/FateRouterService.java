package com.osx.broker.router;


import com.osx.core.router.RouterInfo;
import com.webank.ai.eggroll.api.networking.proxy.Proxy;


public interface FateRouterService {

    RouterInfo route(String srcPartyId, String srcRole, String dstPartyId, String desRole);

    RouterInfo route(Proxy.Packet packet);

    //RouterInfo route(FireworkTransfer.RouteInfo routeInfo);


}
