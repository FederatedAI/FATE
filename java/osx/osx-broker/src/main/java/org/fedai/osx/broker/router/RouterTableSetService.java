package org.fedai.osx.broker.router;

import com.google.common.base.Preconditions;
import com.google.inject.Inject;
import com.google.inject.Singleton;
import org.apache.commons.lang3.StringUtils;
import org.fedai.osx.broker.pojo.RouterTableSetRequest;
import org.fedai.osx.broker.pojo.RouterTableSetResponse;
import org.fedai.osx.broker.service.Register;
import org.fedai.osx.broker.token.TokenValidator;
import org.fedai.osx.broker.token.TokenValidatorRegister;
import org.fedai.osx.core.config.MetaInfo;
import org.fedai.osx.core.constant.ActionType;
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
@Register(uris ={UriConstants.HTTP_SET_ROUTER  ,},allowInterUse = false)
public class RouterTableSetService extends AbstractServiceAdaptorNew<RouterTableSetRequest, RouterTableSetResponse> {

    Logger logger  = LoggerFactory.getLogger(RouterTableSetService.class);
    @Inject
    RouterServiceRegister  routerServiceRegister;

    @Inject
    TokenValidatorRegister tokenValidatorRegister;

    @Override
    protected RouterTableSetResponse doService(OsxContext context, RouterTableSetRequest data) {
        context.setActionType(ActionType.SET_ROUTER.name());
        if(PROPERTY_ROUTER_CHANGE_NEED_TOKEN){
            TokenValidator tokenValidator =  tokenValidatorRegister.select(PROPERTY_ROUTER_CHANGE_TOKEN_VALIDATOR);
            if(tokenValidator!=null) {
                tokenValidator.validate(data.getToken());
            }else {
                logger.error("token validator {} is not found",PROPERTY_ROUTER_CHANGE_TOKEN_VALIDATOR);
            }
        }
        RouterService routerService = routerServiceRegister.select(MetaInfo.PROPERTY_FATE_TECH_PROVIDER);
        Preconditions.checkArgument(data!=null&& StringUtils.isNotEmpty(data.getData()));
        routerService.setRouterTable(data.getData());
        return new  RouterTableSetResponse();
    }

    @Override
    protected  RouterTableSetResponse transformExceptionInfo(OsxContext context, ExceptionInfo exceptionInfo) {
        RouterTableSetResponse routerTableSetResponse = new RouterTableSetResponse();
        routerTableSetResponse.setCode(exceptionInfo.getCode());
        routerTableSetResponse.setMsg(exceptionInfo.getMessage());
        return routerTableSetResponse;
    }
    @Override
    public RouterTableSetRequest decode(Object object) {
        if(object instanceof  String){
            return JsonUtil.json2Object(object.toString(),RouterTableSetRequest.class);
        }
        throw new ParameterException("invalid param");
    }

}
