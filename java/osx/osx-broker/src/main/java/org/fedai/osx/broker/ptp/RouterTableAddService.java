package org.fedai.osx.broker.ptp;

import com.google.inject.Inject;
import com.google.inject.Singleton;
import org.fedai.osx.broker.pojo.*;
import org.fedai.osx.broker.router.RouterService;
import org.fedai.osx.broker.router.RouterServiceRegister;
import org.fedai.osx.broker.service.Register;
import org.fedai.osx.core.constant.*;
import org.fedai.osx.core.context.OsxContext;
import org.fedai.osx.core.exceptions.ExceptionInfo;
import org.fedai.osx.core.router.RouterInfo;
import org.fedai.osx.core.service.AbstractServiceAdaptorNew;
import org.ppc.ptp.Osx;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


@Singleton
@Register(uris ={UriConstants.HTTP_ADD_ROUTER  ,},allowInterUse = false)
public class RouterTableAddService extends AbstractServiceAdaptorNew<RouterAddRequest, RouterAddResponse >{
    Logger logger = LoggerFactory.getLogger(RouterTableAddService.class);
    @Inject
    RouterServiceRegister routerServiceRegister;

    @Override
    protected RouterAddResponse doService(OsxContext context, RouterAddRequest data) {
        context.setActionType(ActionType.SET_ROUTER.name());
        RouterAddResponse  response = new  RouterAddResponse();
        RouterService routerService = routerServiceRegister.getTechProvider(context);
        String allRouterInfo = routerService.addRouterInfo(buildRouterInfo(data));
        response.setData(allRouterInfo);
        return response;
    }
    private  RouterInfo  buildRouterInfo(RouterAddRequest  data){
        RouterInfo  routerInfo =   new RouterInfo();
        routerInfo.setPort(data.getPort());
        routerInfo.setProtocol(data.getProtocol());
        routerInfo.setDesPartyId(data.getDesPartyId());
        routerInfo.setDesRole(data.getDesRole());
        routerInfo.setUrl(data.getUrl());
        routerInfo.setHost(data.getIp());
        routerInfo.setPort(data.getPort());
        routerInfo.setUseSSL(data.isUseSSL());
        routerInfo.setCaFile(data.getCaFile());
        routerInfo.setPrivateKeyFile(data.getPrivateKeyFile());
        routerInfo.setCertChainFile(data.getCertChainFile());
        routerInfo.setUseKeyStore(data.isUseKeyStore());
        routerInfo.setKeyStoreFilePath(data.getKeyStoreFilePath());
        routerInfo.setKeyStorePassword(data.getKeyStorePassword());
        routerInfo.setTrustStoreFilePath(data.getTrustStoreFilePath());
        routerInfo.setTrustStorePassword(data.getTrustStorePassword());
        return  routerInfo;
    }

    @Override
        protected  RouterAddResponse transformExceptionInfo(OsxContext context, ExceptionInfo exceptionInfo) {
        RouterAddResponse consumerResponse = new RouterAddResponse();
        consumerResponse.setCode(exceptionInfo.getCode());
        consumerResponse.setMsg(exceptionInfo.getMessage());
        return consumerResponse;
    }


    @Override
    public RouterAddRequest decode(Object object) {
        return null;
    }

    @Override
    public Osx.Outbound toOutbound(RouterAddResponse response) {
        return null;
    }
}
