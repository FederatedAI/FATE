package com.osx.broker.util;


import com.osx.broker.constants.Direction;
import com.osx.core.router.RouterInfo;


public class ResourceUtil {

//    static public  String  buildResource(Proxy.Metadata metadata){
//        return "";
//    }

    static public String buildResource(RouterInfo routerInfo, Direction direction) {
        return new StringBuilder().append(routerInfo.getResource()).
                append("-").append(direction.name()).toString();
    }

    static public String buildResource(String resource, Direction direction) {
        return new StringBuilder().append(resource).
                append("-").append(direction.name()).toString();
    }


}
