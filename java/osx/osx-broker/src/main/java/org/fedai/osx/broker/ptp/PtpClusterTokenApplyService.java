package org.fedai.osx.broker.ptp;

import com.google.inject.Inject;
import com.google.inject.Singleton;
import com.google.protobuf.ByteString;
import org.fedai.osx.broker.service.Register;
import org.fedai.osx.broker.token.DefaultTokenService;
import org.fedai.osx.core.constant.ActionType;
import org.fedai.osx.core.constant.UriConstants;
import org.fedai.osx.core.context.OsxContext;
import org.fedai.osx.core.exceptions.ExceptionInfo;
import org.fedai.osx.core.service.InboundPackage;
import org.fedai.osx.core.token.TokenRequest;
import org.fedai.osx.core.token.TokenResult;
import org.fedai.osx.core.utils.JsonUtil;
import org.ppc.ptp.Osx;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.charset.StandardCharsets;
@Singleton
//@Register(uri = UriConstants.CLUSTER_TOKEN_APPLY,allowInterUse = false)
public class PtpClusterTokenApplyService  extends AbstractPtpServiceAdaptor< Osx.Inbound, Osx.Outbound> {

    Logger logger = LoggerFactory.getLogger(PtpClusterTokenApplyService.class);
    @Inject
    DefaultTokenService  defaultTokenService;
    @Override
    protected Osx.Outbound doService(OsxContext context, InboundPackage<Osx.Inbound> data) {
        context.setActionType(ActionType.CLUSTER_TOKEN_APPLY.getAlias());
        Osx.Inbound inbound = data.getBody();
        byte[] temp = inbound.getPayload().toByteArray();
        TokenRequest tokenRequest = JsonUtil.json2Object(temp, TokenRequest.class);
        TokenResult tokenResult = defaultTokenService.requestToken(tokenRequest.getResource(),tokenRequest.getAcquireCount(),tokenRequest.isPrioritized());
        Osx.Outbound.Builder  resultBuilder = Osx.Outbound.newBuilder();
        resultBuilder.setPayload(ByteString.copyFrom(JsonUtil.object2Json(tokenResult).getBytes(StandardCharsets.UTF_8)));
        return resultBuilder.build();
    }

    @Override
    protected Osx.Outbound transformExceptionInfo(OsxContext context, ExceptionInfo exceptionInfo) {
        return null;
    }
}
