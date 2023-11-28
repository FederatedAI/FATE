package org.fedai.osx.broker.router;

import com.google.inject.Inject;
import com.google.inject.Singleton;
import org.fedai.osx.broker.pojo.RouterTableGetRequest;
import org.fedai.osx.broker.pojo.RouterTableGetResponse;
import org.fedai.osx.broker.service.Register;
import org.fedai.osx.broker.token.TokenValidator;
import org.fedai.osx.broker.token.TokenValidatorRegister;
import org.fedai.osx.core.config.MetaInfo;
import org.fedai.osx.core.constant.ActionType;
import org.fedai.osx.core.constant.StatusCode;
import org.fedai.osx.core.constant.UriConstants;
import org.fedai.osx.core.context.OsxContext;
import org.fedai.osx.core.exceptions.ExceptionInfo;
import org.fedai.osx.core.exceptions.ParameterException;
import org.fedai.osx.core.service.AbstractServiceAdaptorNew;
import org.fedai.osx.core.utils.JsonUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static org.fedai.osx.core.config.MetaInfo.PROPERTY_ROUTER_CHANGE_NEED_TOKEN;
import static org.fedai.osx.core.config.MetaInfo.PROPERTY_ROUTER_CHANGE_TOKEN_VALIDATOR;

@Singleton
@Register(uris ={UriConstants.HTTP_GET_ROUTER  ,},allowInterUse = false)
public class RouterTableGetService extends AbstractServiceAdaptorNew<RouterTableGetRequest, RouterTableGetResponse>
    {
        Logger logger  = LoggerFactory.getLogger(RouterTableGetService.class);
        @Inject
        RouterServiceRegister routerServiceRegister;
        @Inject
        TokenValidatorRegister  tokenValidatorRegister;

        @Override
        protected RouterTableGetResponse doService(OsxContext context, RouterTableGetRequest data) {
            context.setActionType(ActionType.GET_ROUTER.name());
            if(PROPERTY_ROUTER_CHANGE_NEED_TOKEN){
                TokenValidator tokenValidator =  tokenValidatorRegister.select(PROPERTY_ROUTER_CHANGE_TOKEN_VALIDATOR);
                if(tokenValidator!=null) {
                    tokenValidator.validate(data.getToken());
                }else {
                    logger.error("token validator {} is not found",PROPERTY_ROUTER_CHANGE_TOKEN_VALIDATOR);
                }
            }
        RouterService routerService = routerServiceRegister.select(MetaInfo.PROPERTY_FATE_TECH_PROVIDER);
        RouterTableGetResponse response =     new  RouterTableGetResponse();
        response.setContent(routerService.getRouterTable());
        response.setCode(StatusCode.PTP_SUCCESS);
        return response;
        }

        @Override
        protected  RouterTableGetResponse transformExceptionInfo(OsxContext context, ExceptionInfo exceptionInfo) {
            RouterTableGetResponse response = new RouterTableGetResponse();
            response.setCode(exceptionInfo.getCode());
            response.setMsg(exceptionInfo.getMessage());
        return response;
    }
        @Override
        public RouterTableGetRequest decode(Object object) {
        if(object instanceof  String){
            RouterTableGetRequest result =  JsonUtil.json2Object(object.toString(),RouterTableGetRequest.class);
            if(result==null){
                throw new ParameterException("invalid param for router operation");
            }
            return  result;
        }
        throw new ParameterException("invalid param");
    }

    }
