package org.fedai.osx.broker.interceptor;


import com.webank.ai.eggroll.api.networking.proxy.Proxy;
import org.fedai.osx.api.context.Context;
import org.fedai.osx.broker.util.TransferUtil;
import org.fedai.osx.core.service.InboundPackage;
import org.fedai.osx.core.service.Interceptor;
import org.fedai.osx.core.service.OutboundPackage;

public class UnaryCallHandleInterceptor implements Interceptor<Context, Proxy.Packet, Proxy.Packet> {

    @Override
    public void doProcess(Context context, InboundPackage<Proxy.Packet> inboundPackage, OutboundPackage<Proxy.Packet> outboundPackage) throws Exception {
        Proxy.Packet packet = inboundPackage.getBody();
        TransferUtil.assableContextFromProxyPacket(context, packet);
    }
}
