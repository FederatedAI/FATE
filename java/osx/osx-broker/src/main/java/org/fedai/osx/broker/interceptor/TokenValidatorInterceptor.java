package org.fedai.osx.broker.interceptor;

import com.google.inject.Inject;
import com.google.inject.Singleton;
import org.fedai.osx.broker.security.TokenValidator;
import org.fedai.osx.broker.security.TokenValidatorRegister;
import org.fedai.osx.core.config.MetaInfo;
import org.fedai.osx.core.constant.Dict;
import org.fedai.osx.core.context.OsxContext;
import org.fedai.osx.core.service.InboundPackage;
import org.fedai.osx.core.service.Interceptor;
import org.fedai.osx.core.service.OutboundPackage;
@Singleton
public class TokenValidatorInterceptor implements Interceptor< InboundPackage, OutboundPackage> {
    @Inject
    TokenValidatorRegister  tokenValidatorRegister;
    @Override
    public void doProcess(OsxContext context, InboundPackage inboundPackage, OutboundPackage outboundPackage) throws Exception {
        if (MetaInfo.PROPERTY_OPEN_TOKEN_VALIDATOR) {
            TokenValidator tokenValidator = tokenValidatorRegister.getTokenValidator(getValidatorKey(context), Dict.DEFAULT);
            if (tokenValidator != null) {
                tokenValidator.validate(context, context.getToken());
            }
        }
    }

    private String getValidatorKey(OsxContext context) {
        String srcPartyId = context.getSrcNodeId();
        return srcPartyId;
    }
}
