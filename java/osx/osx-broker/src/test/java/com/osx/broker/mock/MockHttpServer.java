package com.osx.broker.mock;

import com.osx.broker.ServiceContainer;
import com.osx.broker.server.OsxServer;
import com.osx.core.config.MetaInfo;

import java.util.HashSet;


public class MockHttpServer {





    public  static void main(String[] args){
        HashSet selfPartyIds =  new HashSet();
        selfPartyIds.add("10001");
        MetaInfo.PROPERTY_SELF_PARTY= selfPartyIds;
        MetaInfo.PROPERTY_GRPC_PORT=9372;
        MetaInfo.PROPERTY_HTTP_PORT=8222;
        MetaInfo.PROPERTY_OPEN_HTTP_SERVER = Boolean.TRUE;
        ServiceContainer.init();
    }


}
