package com.osx.broker.interceptor;


import com.osx.core.context.Context;
import com.osx.core.service.InboundPackage;
import com.osx.core.service.Interceptor;
import com.osx.core.service.OutboundPackage;
import org.ppc.ptp.Pcp;

public class PpcRouterInterceptor implements Interceptor<Pcp.Inbound,Pcp.Outbound> {

    public void doPreProcess(Context context, InboundPackage<Pcp.Inbound> inboundPackage, OutboundPackage<Pcp.Outbound> outboundPackage) throws Exception {
        


    }

}
