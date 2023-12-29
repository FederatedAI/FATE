package org.fedai.osx.broker.router;
import org.fedai.osx.core.router.RouterInfo;

import java.util.Set;

public interface RouterService {

    public RouterInfo route(String srcPartyId, String srcRole, String dstPartyId, String desRole);

    public String addRouterInfo(RouterInfo  routerInfo);

    public void  setRouterTable(String  content);

    public String getRouterTable();

    public void  setSelfPartyIds(Set<String> partyIds);

}
