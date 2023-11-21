package org.fedai.osx.broker.ptp;

import com.google.protobuf.ByteString;
import com.webank.ai.eggroll.api.networking.proxy.Proxy;
import io.grpc.stub.StreamObserver;
import lombok.extern.slf4j.Slf4j;
import org.fedai.osx.broker.constants.MessageFlag;
import org.fedai.osx.broker.router.DefaultFateRouterServiceImpl;
import org.fedai.osx.broker.router.RouterService;
import org.fedai.osx.broker.util.TransferUtil;
import org.fedai.osx.core.constant.Dict;
import org.fedai.osx.core.constant.TransferStatus;
import org.fedai.osx.core.constant.UriConstants;
import org.fedai.osx.core.context.OsxContext;
import org.fedai.osx.core.exceptions.ExceptionInfo;
import org.fedai.osx.core.router.RouterInfo;
import org.ppc.ptp.Osx;

import java.nio.charset.StandardCharsets;

@Slf4j
public class DirectBackStreamObserver implements StreamObserver<Proxy.Metadata> {

    TransferStatus transferStatus = TransferStatus.INIT;
    RouterService defaultFateRouterService;
    OsxContext context;
    String oriTopic;
    String backTopic;
    String sessionId;
    String srcNodeId;
    String desNodeId;
    RouterInfo revertRouterInfo;
    DirectBackStreamObserver(RouterService defaultFateRouterService,
                             String oriTopic, String sessionId, String srcNodeId, String desNodeId) {
        this.defaultFateRouterService = defaultFateRouterService;
        this.oriTopic = oriTopic;
        this.sessionId = sessionId;
        this.srcNodeId = srcNodeId;
        this.desNodeId = desNodeId;
    }

    private void init() {
        context = new OsxContext();
        context.setSessionId(sessionId);
        context.setUri(UriConstants.PUSH);
        context.setSrcNodeId(srcNodeId);
        context.setDesNodeId(desNodeId);
        revertRouterInfo = defaultFateRouterService.route(srcNodeId,"",desNodeId , "" );
        transferStatus = TransferStatus.TRANSFERING;
        backTopic = buildBackTopic(oriTopic);
    }

    private String buildBackTopic(String oriTopic) {
        backTopic = oriTopic.replaceAll(Dict.STREAM_SEND_TOPIC_PREFIX, Dict.STREAM_BACK_TOPIC_PREFIX);
        return backTopic;
    }

    @Override
    public void onNext(Proxy.Metadata metadata) {
        //将其对调后再查路由

        try {
            if (transferStatus.equals(TransferStatus.INIT)) {
                init();
            }
            OsxContext.pushThreadLocalContext(context);
            context.setMessageFlag(MessageFlag.SENDMSG.name());
//        Osx.Inbound.Builder  inboundBuilder = TransferUtil.buildInbound(provider,desPartyId,srcPartyId, TargetMethod.PRODUCE_MSG.name(),
//                backTopic, MessageFlag.SENDMSG,sessionId, metadata.toByteString().toByteArray());
            Osx.Inbound.Builder inboundBuilder = Osx.Inbound.newBuilder();
            Osx.PushInbound.Builder pushBuilder = Osx.PushInbound.newBuilder();
            pushBuilder.setPayload(metadata.toByteString());
            pushBuilder.setTopic(backTopic);
            inboundBuilder.setPayload(pushBuilder.build().toByteString());
            TransferUtil.redirect(context, inboundBuilder.build(), revertRouterInfo, true);
        } finally {
            OsxContext.popThreadLocalContext();
        }
    }

    @Override
    public void onError(Throwable throwable) {
        try {

            OsxContext.pushThreadLocalContext(context);
            ExceptionInfo exceptionInfo = new ExceptionInfo();
            exceptionInfo.setMessage(throwable.getMessage());
            String message = throwable.getMessage();
            context.setMessageFlag(MessageFlag.ERROR.name());
            Osx.Inbound.Builder inboundBuilder = Osx.Inbound.newBuilder();
            Osx.PushInbound.Builder pushBuilder = Osx.PushInbound.newBuilder();
            pushBuilder.setPayload(ByteString.copyFrom(exceptionInfo.toString().getBytes(StandardCharsets.UTF_8)));
            pushBuilder.setTopic(backTopic);
            inboundBuilder.setPayload(pushBuilder.build().toByteString());
            TransferUtil.redirect(context, inboundBuilder.build(), revertRouterInfo, true);
        } finally {
            OsxContext.popThreadLocalContext();
        }

    }

    @Override
    public void onCompleted() {
        /**
         * 完成回调
         */
        try {
            OsxContext.pushThreadLocalContext(context);
            context.setMessageFlag(MessageFlag.COMPELETED.name());
            Osx.Inbound.Builder inboundBuilder = Osx.Inbound.newBuilder();
            Osx.PushInbound.Builder pushBuilder = Osx.PushInbound.newBuilder();
            pushBuilder.setPayload(ByteString.copyFrom("onCompleted".getBytes(StandardCharsets.UTF_8)));
            pushBuilder.setTopic(backTopic);
            inboundBuilder.setPayload(pushBuilder.build().toByteString());
            Osx.Outbound result = (Osx.Outbound) TransferUtil.redirect(context, inboundBuilder.build(), revertRouterInfo, true);
        } catch (Exception e) {
            log.error("receive completed error", e);
        } finally {
            OsxContext.popThreadLocalContext();
        }
    }
}
