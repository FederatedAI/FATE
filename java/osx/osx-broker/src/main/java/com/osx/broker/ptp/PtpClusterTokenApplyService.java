package com.osx.broker.ptp;

import com.google.protobuf.ByteString;
import com.osx.api.context.Context;
import com.osx.broker.ServiceContainer;
import com.osx.core.config.MetaInfo;
import com.osx.core.constant.ActionType;
import com.osx.core.context.FateContext;
import com.osx.core.exceptions.RemoteRpcException;
import com.osx.core.flow.*;
import com.osx.core.frame.GrpcConnectionFactory;
import com.osx.core.service.InboundPackage;
import com.osx.core.token.TokenRequest;
import com.osx.core.token.TokenResult;
import com.osx.core.token.TokenResultStatus;
import com.osx.core.utils.JsonUtil;
import io.grpc.ManagedChannel;
import org.apache.commons.lang3.StringUtils;
import org.ppc.ptp.Osx;
import org.ppc.ptp.PrivateTransferProtocolGrpc;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.charset.StandardCharsets;

public class PtpClusterTokenApplyService  extends AbstractPtpServiceAdaptor {

    Logger logger = LoggerFactory.getLogger(PtpClusterTokenApplyService.class);
    @Override
    protected Osx.Outbound doService(FateContext context, InboundPackage<Osx.Inbound> data) {
        context.setActionType(ActionType.CLUSTER_TOKEN_APPLY.getAlias());
        Osx.Inbound inbound = data.getBody();
        byte[] temp = inbound.getPayload().toByteArray();
        TokenRequest tokenRequest = JsonUtil.json2Object(temp, TokenRequest.class);
        TokenResult  tokenResult = ServiceContainer.defaultTokenService.requestToken(tokenRequest.getResource(),tokenRequest.getAcquireCount(),tokenRequest.isPrioritized());
        Osx.Outbound.Builder  resultBuilder = Osx.Outbound.newBuilder();
        resultBuilder.setPayload(ByteString.copyFrom(JsonUtil.object2Json(tokenResult).getBytes(StandardCharsets.UTF_8)));
        return resultBuilder.build();
    }
}
