package com.osx.broker.interceptor;


import com.osx.core.context.Context;
import com.osx.core.service.InboundPackage;
import com.osx.core.service.Interceptor;
import com.osx.core.service.OutboundPackage;
import org.ppc.ptp.Osx;


public class PpcRouterInterceptor implements Interceptor<Osx.Inbound, Osx.Outbound> {

    public void doPreProcess(Context context, InboundPackage<Osx.Inbound> inboundPackage, OutboundPackage<Osx.Outbound> outboundPackage) throws Exception {


    }

}
