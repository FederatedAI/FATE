package org.fedai.osx.broker.router;

import com.google.inject.Inject;
import com.google.inject.Singleton;
import org.fedai.osx.broker.pojo.RouterAddRequest;
import org.fedai.osx.broker.pojo.RouterAddResponse;
import org.fedai.osx.broker.service.Register;
import org.fedai.osx.broker.token.TokenValidator;
import org.fedai.osx.broker.token.TokenValidatorRegister;
import org.fedai.osx.core.constant.ActionType;
import org.fedai.osx.core.constant.UriConstants;
import org.fedai.osx.core.context.OsxContext;
import org.fedai.osx.core.exceptions.ExceptionInfo;
import org.fedai.osx.core.exceptions.ParameterException;
import org.fedai.osx.core.router.RouterInfo;
import org.fedai.osx.core.service.AbstractServiceAdaptorNew;
import org.fedai.osx.core.utils.JsonUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static org.fedai.osx.core.config.MetaInfo.PROPERTY_ROUTER_CHANGE_NEED_TOKEN;
import static org.fedai.osx.core.config.MetaInfo.PROPERTY_ROUTER_CHANGE_TOKEN_VALIDATOR;


@Singleton
@Register(uris ={UriConstants.HTTP_ADD_ROUTER  ,},allowInterUse = false)
public class RouterTableAddService extends AbstractServiceAdaptorNew<RouterAddRequest, RouterAddResponse >{

    Logger logger = LoggerFactory.getLogger(RouterTableAddService.class);
    @Inject
    RouterServiceRegister routerServiceRegister;
    @Inject
    TokenValidatorRegister tokenValidatorRegister;

    @Override
    protected RouterAddResponse doService(OsxContext context, RouterAddRequest data) {
        context.setActionType(ActionType.ADD_ROUTER.name());
        if(PROPERTY_ROUTER_CHANGE_NEED_TOKEN){
            TokenValidator tokenValidator =  tokenValidatorRegister.select(PROPERTY_ROUTER_CHANGE_TOKEN_VALIDATOR);
            if(tokenValidator!=null) {
                tokenValidator.validate(data.getToken());
            }else {
                logger.error("token validator {} is not found",PROPERTY_ROUTER_CHANGE_TOKEN_VALIDATOR);
            }
        }
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
        if(object instanceof  String){
            return JsonUtil.json2Object(object.toString(),RouterAddRequest.class);
        }
        throw new ParameterException("invalid param for op");
    }


}
