package org.fedai.osx.broker.service;

import com.webank.ai.eggroll.api.networking.proxy.Proxy;

import org.fedai.osx.broker.router.DefaultFateRouterServiceImpl;
import org.fedai.osx.core.context.OsxContext;
import org.fedai.osx.core.exceptions.ExceptionInfo;
import org.fedai.osx.core.service.AbstractServiceAdaptor;
import org.fedai.osx.core.service.InboundPackage;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class RouteService extends AbstractServiceAdaptor< Proxy.Packet, Proxy.Packet> {

    Logger logger = LoggerFactory.getLogger(RouteService.class);

    @Override
    protected Proxy.Packet doService(OsxContext context, InboundPackage<Proxy.Packet> data) {
        DefaultFateRouterServiceImpl defaultFateRouterService = new DefaultFateRouterServiceImpl();
        defaultFateRouterService.saveRouterTable(context, data);
        Proxy.Packet.Builder resultBuilder = Proxy.Packet.newBuilder();
        return resultBuilder.build();
    }

    @Override
    protected Proxy.Packet transformExceptionInfo(OsxContext context, ExceptionInfo exceptionInfo) {
        return null;
    }
}
