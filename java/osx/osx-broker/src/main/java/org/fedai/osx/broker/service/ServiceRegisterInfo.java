package org.fedai.osx.broker.service;

import lombok.Data;
import org.fedai.osx.broker.constants.ServiceType;
import org.fedai.osx.core.context.Protocol;
import org.fedai.osx.core.router.RouterInfo;
import org.fedai.osx.core.service.AbstractServiceAdaptorNew;

@Data
public class ServiceRegisterInfo {
    String uri;
    ServiceType serviceType;
    Protocol protocol;
//    String host;
//    String ip;
//    int port;

    RouterInfo routerInfo;
    String nodeId;
    boolean  allowInterUse;
    AbstractServiceAdaptorNew serviceAdaptor;


    public String  buildRegisterKey(){

        return buildKey(nodeId,uri);

    }

    public  static  String   buildKey(String nodeId,String  uri){
        StringBuilder  sb = new StringBuilder();
        sb.append(nodeId).append(":").append(uri);
        return sb.toString();
    }
}
