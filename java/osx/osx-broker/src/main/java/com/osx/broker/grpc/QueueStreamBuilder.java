package com.osx.broker.grpc;

import com.google.protobuf.*;
import com.osx.api.router.RouterInfo;
import com.osx.broker.ServiceContainer;
import com.osx.broker.constants.MessageFlag;
import com.osx.broker.consumer.SourceGrpcEventHandler;
import com.osx.broker.eggroll.PushEventHandler;
import com.osx.broker.queue.CreateQueueResult;
import com.osx.broker.queue.TransferQueue;
import com.osx.broker.util.TransferUtil;
import com.osx.core.config.MetaInfo;
import com.osx.core.constant.ActionType;
import com.osx.core.constant.Dict;
import com.osx.core.constant.StatusCode;

import com.osx.core.context.FateContext;
import com.osx.core.exceptions.ErrorMessageUtil;
import com.osx.core.exceptions.ExceptionInfo;
import com.osx.core.exceptions.RemoteRpcException;

import com.osx.core.frame.GrpcConnectionFactory;
import com.osx.core.ptp.TargetMethod;
import com.osx.core.utils.JsonUtil;
import com.webank.ai.eggroll.api.networking.proxy.Proxy;
import io.grpc.ManagedChannel;
import io.grpc.stub.StreamObserver;
import org.ppc.ptp.Osx;
import org.ppc.ptp.PrivateTransferProtocolGrpc;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Random;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CountDownLatch;
import java.nio.charset.StandardCharsets;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicInteger;


public class QueueStreamBuilder {


    ConcurrentHashMap backRegister = new ConcurrentHashMap() ;



    /**
     *  在流的开端调用
     * @param respStreamObserver
     * @param parser
     * @param srcPartyId
     * @param desPartyId
     * @param sessionId
     * @return
     */

    private  static AtomicInteger  count= new AtomicInteger(0);

    private  static Logger logger = LoggerFactory.getLogger(QueueStreamBuilder.class);
    public static  StreamObserver createStreamFromOrigin(FateContext context ,
                                                         StreamObserver respStreamObserver,
                                                         Parser parser,
                                                         RouterInfo routerInfo,
                                                         String srcPartyId,
                                                         String desPartyId,
                                                         String sessionId,
                                                         CountDownLatch countDownLatch
                               ){

        //String uuid = UUID.randomUUID().toString();
        int  temp = count.addAndGet(1);
        long  now = System.currentTimeMillis();
        //srcPartyId+"_"+desPartyId
        String backTopic = Dict.STREAM_BACK_TOPIC_PREFIX +now+ "_"+sessionId+"_"+temp;
        String sendTopic = Dict.STREAM_SEND_TOPIC_PREFIX +now+"_"+sessionId+"_"+temp;
        context.setTopic(sendTopic);
        context.setActionType(ActionType.MSG_REDIRECT.getAlias());
        CreateQueueResult createQueueResult = ServiceContainer.transferQueueManager.createNewQueue(backTopic, sessionId, true);
        if (createQueueResult.getTransferQueue() == null) {
            throw new RemoteRpcException("create queue error");
        }
        TransferQueue answerQueue = createQueueResult.getTransferQueue();
        ServiceContainer.consumerManager.createEventDrivenConsumer(backTopic,new SourceGrpcEventHandler(respStreamObserver,parser));
        StreamObserver forwardPushReqSO = new StreamObserver<AbstractMessage>() {

            @Override
            public void onNext(AbstractMessage message) {
                try {
                    context.setMessageFlag(MessageFlag.SENDMSG.name());
                    Osx.Inbound.Builder inboundBuilder = TransferUtil.buildInbound(MetaInfo.PROPERTY_FATE_TECH_PROVIDER, srcPartyId, desPartyId, TargetMethod.PRODUCE_MSG.name(), sendTopic, MessageFlag.SENDMSG, sessionId, message.toByteArray());
                    Osx.Outbound outbound = TransferUtil.redirect(context, inboundBuilder.build(), routerInfo, true);
                    TransferUtil.checkResponse(outbound);
                }catch(Exception e){
                    throw ErrorMessageUtil.toGrpcRuntimeException(e);
                }
            }

            @Override
            public void onError(Throwable throwable) {
                try {
                    context.setMessageFlag(MessageFlag.ERROR.name());
                    ExceptionInfo exceptionInfo = new ExceptionInfo();
                    exceptionInfo.setMessage(throwable.getMessage());
                    String errorData = JsonUtil.object2Json(exceptionInfo);
                    Osx.Inbound.Builder inboundBuilder = TransferUtil.buildInbound(MetaInfo.PROPERTY_FATE_TECH_PROVIDER, srcPartyId, desPartyId, TargetMethod.PRODUCE_MSG.name(),
                                    sendTopic, MessageFlag.ERROR, sessionId, errorData.getBytes(StandardCharsets.UTF_8))
                            .putMetadata(Osx.Metadata.MessageFlag.name(), MessageFlag.ERROR.name());
                    Osx.Outbound outbound = TransferUtil.redirect(context, inboundBuilder.build(), routerInfo, true);
                    TransferUtil.checkResponse(outbound);
                    countDownLatch.countDown();
                }catch (Exception e){
                    throw ErrorMessageUtil.toGrpcRuntimeException(e);
                }
            }

            @Override
            public void onCompleted() {
                try {
                    context.setMessageFlag(MessageFlag.COMPELETED.name());
                    Osx.Inbound.Builder inboundBuilder = TransferUtil.buildInbound(MetaInfo.PROPERTY_FATE_TECH_PROVIDER, srcPartyId, desPartyId, TargetMethod.PRODUCE_MSG.name(),
                                    sendTopic, MessageFlag.COMPELETED, sessionId, "completed".getBytes(StandardCharsets.UTF_8))
                            .putMetadata(Osx.Metadata.MessageFlag.name(), MessageFlag.COMPELETED.name());
                    Osx.Outbound outbound = TransferUtil.redirect(context, inboundBuilder.build(), routerInfo, true);

                    TransferUtil.checkResponse(outbound);
                    countDownLatch.countDown();

                } catch (Exception e) {
                    throw ErrorMessageUtil.toGrpcRuntimeException(e);
                }
            }
        };
        return forwardPushReqSO;

    };


}
