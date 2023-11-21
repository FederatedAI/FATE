package org.fedai.osx.broker.router;


import com.webank.ai.eggroll.api.networking.proxy.Proxy;
import org.fedai.osx.core.router.RouterInfo;

public interface RouterService {

    public RouterInfo route(String srcPartyId, String srcRole, String dstPartyId, String desRole);

    public String addRouterInfo(RouterInfo  routerInfo);

    public String  getAllRouterInfo();

}
