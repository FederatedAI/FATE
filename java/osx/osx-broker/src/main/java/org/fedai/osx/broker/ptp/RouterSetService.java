package org.fedai.osx.broker.ptp;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.google.common.base.Preconditions;
import com.google.inject.Inject;
import com.google.inject.Singleton;
import io.grpc.ManagedChannel;
import io.grpc.stub.StreamObserver;
import org.apache.commons.lang3.StringUtils;
import org.fedai.osx.broker.consumer.ConsumerManager;
import org.fedai.osx.broker.consumer.UnaryConsumer;
import org.fedai.osx.broker.pojo.*;
import org.fedai.osx.broker.queue.*;
import org.fedai.osx.broker.router.RouterService;
import org.fedai.osx.broker.router.RouterServiceRegister;
import org.fedai.osx.broker.service.Register;
import org.fedai.osx.core.config.MetaInfo;
import org.fedai.osx.core.constant.*;
import org.fedai.osx.core.context.OsxContext;
import org.fedai.osx.core.context.Protocol;
import org.fedai.osx.core.exceptions.ExceptionInfo;
import org.fedai.osx.core.exceptions.TransferQueueNotExistException;
import org.fedai.osx.core.frame.GrpcConnectionFactory;
import org.fedai.osx.core.router.RouterInfo;
import org.fedai.osx.core.service.AbstractServiceAdaptorNew;
import org.fedai.osx.core.utils.JsonUtil;
import org.ppc.ptp.Osx;
import org.ppc.ptp.PrivateTransferTransportGrpc;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


@Singleton
@Register(uris ={UriConstants.HTTP_CHANGE_ROUTER},allowInterUse = false)
public class RouterSetService extends AbstractServiceAdaptorNew<RouterSetRequest, RouterSetResponse>{

    Logger logger = LoggerFactory.getLogger(RouterSetService.class);
    @Inject
    RouterServiceRegister routerServiceRegister;

    @Override
    protected RouterSetResponse doService(OsxContext context, RouterSetRequest data) {
        context.setActionType(ActionType.SET_ROUTER.name());
        RouterSetResponse  response = new  RouterSetResponse();
        RouterService routerService = routerServiceRegister.getTechProvider(context);
        String allRouterInfo = routerService.addRouterInfo(buildRouterInfo(data));
        response.setData(allRouterInfo);
        return response;
    }
    private  RouterInfo  buildRouterInfo(RouterSetRequest  data){
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
        protected  RouterSetResponse transformExceptionInfo(OsxContext context, ExceptionInfo exceptionInfo) {
        RouterSetResponse consumerResponse = new RouterSetResponse();
        consumerResponse.setCode(exceptionInfo.getCode());
        consumerResponse.setMsg(exceptionInfo.getMessage());
        return consumerResponse;
    }


    @Override
    public RouterSetRequest decode(Object object) {
        return null;
    }

    @Override
    public Osx.Outbound toOutbound(RouterSetResponse response) {
        return null;
    }
}
