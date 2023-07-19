package com.osx.broker.service;

import com.osx.api.context.Context;
import com.osx.broker.router.DefaultFateRouterServiceImpl;
import com.osx.core.exceptions.ExceptionInfo;
import com.osx.core.service.AbstractServiceAdaptor;
import com.osx.core.service.InboundPackage;
import com.webank.ai.eggroll.api.networking.proxy.Proxy;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class RouteService extends AbstractServiceAdaptor<Context, Proxy.Packet, Proxy.Packet> {

    Logger logger = LoggerFactory.getLogger(RouteService.class);

    @Override
    protected Proxy.Packet doService(Context context, InboundPackage<Proxy.Packet> data) {
        DefaultFateRouterServiceImpl defaultFateRouterService = new DefaultFateRouterServiceImpl();
        defaultFateRouterService.saveRouterTable(context, data);
        Proxy.Packet.Builder resultBuilder = Proxy.Packet.newBuilder();
        return resultBuilder.build();
    }

    @Override
    protected Proxy.Packet transformExceptionInfo(Context context, ExceptionInfo exceptionInfo) {
        return null;
    }
}
