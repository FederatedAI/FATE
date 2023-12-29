package org.fedai.osx.broker.service;

import lombok.Data;
import org.apache.commons.lang3.StringUtils;
import org.fedai.osx.broker.constants.ServiceType;
import org.fedai.osx.core.context.Protocol;
import org.fedai.osx.core.router.RouterInfo;
import org.fedai.osx.core.service.AbstractServiceAdaptorNew;
import org.fedai.osx.core.utils.UrlUtil;

@Data
public class ServiceRegisterInfo {
    String serviceId;
    String uri;
    ServiceType serviceType;
    Protocol protocol;
    RouterInfo routerInfo;
    String nodeId;
    boolean allowInterUse;
    AbstractServiceAdaptorNew serviceAdaptor;

    public static String buildKey(String nodeId, String uri) {
        StringBuilder sb = new StringBuilder();
        if(StringUtils.isNotEmpty(nodeId))
        sb.append(nodeId) .append(":");
        sb.append(UrlUtil.parseUri(uri));
        return sb.toString();
    }

    public String buildRegisterKey() {
        return buildKey(nodeId, uri);
    }
}
