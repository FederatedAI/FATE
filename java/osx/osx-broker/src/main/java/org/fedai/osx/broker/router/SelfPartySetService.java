package org.fedai.osx.broker.router;

import com.google.common.base.Preconditions;
import com.google.inject.Singleton;
import org.fedai.osx.broker.pojo.SetSelfPartyRequest;
import org.fedai.osx.broker.pojo.SetSelfPartyResponse;
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

import javax.inject.Inject;

import static org.fedai.osx.core.config.MetaInfo.PROPERTY_ROUTER_CHANGE_NEED_TOKEN;
import static org.fedai.osx.core.config.MetaInfo.PROPERTY_ROUTER_CHANGE_TOKEN_VALIDATOR;

@Singleton
@Register(uris ={UriConstants.HTTP_SET_SELF  },allowInterUse = false)
public class SelfPartySetService extends AbstractServiceAdaptorNew<SetSelfPartyRequest, SetSelfPartyResponse> {

    Logger logger  = LoggerFactory.getLogger(SelfPartySetService.class);
    @Inject
    RouterServiceRegister  routerServiceRegister;
    @Inject
    TokenValidatorRegister tokenValidatorRegister;

    @Override
    protected SetSelfPartyResponse doService(OsxContext context, SetSelfPartyRequest data) {
        context.setActionType(ActionType.SET_SELF_PARTY.name());
        if(PROPERTY_ROUTER_CHANGE_NEED_TOKEN){
            TokenValidator tokenValidator =  tokenValidatorRegister.select(PROPERTY_ROUTER_CHANGE_TOKEN_VALIDATOR);
            if(tokenValidator!=null) {
                tokenValidator.validate(data.getToken());
            }else {
                logger.error("token validator {} is not found",PROPERTY_ROUTER_CHANGE_TOKEN_VALIDATOR);
            }
        }
        Preconditions.checkArgument(data!=null&&data.getSelfParty()!=null,"self_party is null");

        RouterService routerService =routerServiceRegister.select(MetaInfo.PROPERTY_FATE_TECH_PROVIDER);
        routerService.setSelfPartyIds(data.getSelfParty());
        SetSelfPartyResponse  response =  new SetSelfPartyResponse();
        response.setCode(StatusCode.SUCCESS);
        return response;
    }

    @Override
    protected SetSelfPartyResponse transformExceptionInfo(OsxContext context, ExceptionInfo exceptionInfo) {
        SetSelfPartyResponse  response =  new SetSelfPartyResponse();
        response.setCode(exceptionInfo.getCode());
        response.setMsg(exceptionInfo.getMessage());
        return  response;
    }

    @Override
    public SetSelfPartyRequest decode(Object object) {
        if(object instanceof  String){
           return  JsonUtil.json2Object(object.toString(),SetSelfPartyRequest.class);
        }
        throw new ParameterException("invalid param for set self party");
    }
}






