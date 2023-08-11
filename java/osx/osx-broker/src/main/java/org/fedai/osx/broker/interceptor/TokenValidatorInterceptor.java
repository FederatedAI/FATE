package org.fedai.osx.broker.interceptor;

import org.fedai.osx.api.context.Context;
import org.fedai.osx.broker.ServiceContainer;
import org.fedai.osx.broker.security.TokenValidator;
import org.fedai.osx.core.config.MetaInfo;
import org.fedai.osx.core.constant.Dict;
import org.fedai.osx.core.service.InboundPackage;
import org.fedai.osx.core.service.Interceptor;
import org.fedai.osx.core.service.OutboundPackage;

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
