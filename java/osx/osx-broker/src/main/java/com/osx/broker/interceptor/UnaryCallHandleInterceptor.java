package com.osx.broker.interceptor;


import com.osx.api.context.Context;
import com.osx.core.service.InboundPackage;
import com.osx.core.service.Interceptor;
import com.osx.core.service.OutboundPackage;
import com.webank.ai.eggroll.api.networking.proxy.Proxy;

import static com.osx.broker.util.TransferUtil.assableContextFromProxyPacket;

public class UnaryCallHandleInterceptor implements Interceptor<Context, Proxy.Packet, Proxy.Packet> {

    @Override
    public void doProcess(Context context, InboundPackage<Proxy.Packet> inboundPackage, OutboundPackage<Proxy.Packet> outboundPackage) throws Exception {
        Proxy.Packet packet = inboundPackage.getBody();
        assableContextFromProxyPacket(context, packet);
    }
}
