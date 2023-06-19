package com.osx.broker.interceptor;

import com.osx.api.context.Context;
import com.osx.broker.ServiceContainer;
import com.osx.broker.security.TokenValidator;
import com.osx.core.config.MetaInfo;
import com.osx.core.constant.Dict;
import com.osx.core.service.InboundPackage;
import com.osx.core.service.Interceptor;
import com.osx.core.service.OutboundPackage;

public class TokenValidatorInterceptor implements Interceptor<Context, InboundPackage, OutboundPackage> {

    @Override
    public void doProcess(Context context, InboundPackage inboundPackage, OutboundPackage outboundPackage) throws Exception {
        if (MetaInfo.PROPERTY_OPEN_TOKEN_VALIDATOR) {
            TokenValidator tokenValidator = ServiceContainer.tokenValidatorRegister.getTokenValidator(getValidatorKey(context), Dict.DEFAULT);
            if (tokenValidator != null) {
                tokenValidator.validate(context, context.getToken());
            }
        }
    }

    private String getValidatorKey(Context context) {
        String srcPartyId = context.getSrcPartyId();
        return srcPartyId;
    }
}
