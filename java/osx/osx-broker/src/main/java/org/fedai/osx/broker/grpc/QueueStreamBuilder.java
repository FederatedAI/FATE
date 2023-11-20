package org.fedai.osx.broker.grpc;

import com.google.protobuf.AbstractMessage;
import com.google.protobuf.ByteString;
import com.google.protobuf.InvalidProtocolBufferException;
import com.google.protobuf.Parser;
import com.webank.ai.eggroll.api.networking.proxy.Proxy;
import io.grpc.stub.StreamObserver;
import org.fedai.osx.broker.constants.MessageFlag;
import org.fedai.osx.broker.consumer.ConsumerManager;
//import org.fedai.osx.broker.consumer.SourceGrpcEventHandler;
import org.fedai.osx.broker.queue.*;
import org.fedai.osx.broker.util.TransferUtil;
import org.fedai.osx.core.config.MetaInfo;
import org.fedai.osx.core.constant.ActionType;
import org.fedai.osx.core.constant.Dict;
import org.fedai.osx.core.constant.QueueType;
import org.fedai.osx.core.context.OsxContext;
import org.fedai.osx.core.exceptions.ErrorMessageUtil;
import org.fedai.osx.core.exceptions.ExceptionInfo;
import org.fedai.osx.core.exceptions.RemoteRpcException;
import org.fedai.osx.core.ptp.TargetMethod;
import org.fedai.osx.core.router.RouterInfo;
import org.fedai.osx.core.utils.JsonUtil;
import org.ppc.ptp.Osx;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.charset.StandardCharsets;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicInteger;


public class QueueStreamBuilder {
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
    public static  StreamObserver createStreamFromOrigin(OsxContext context ,
                                                         TransferQueueManager transferQueueManager,
                                                         StreamObserver respStreamObserver,
                                                         Parser parser,
                                                         RouterInfo routerInfo,
                                                         String srcPartyId,
                                                         String desPartyId,
                                                         String sessionId,
                                                         CountDownLatch countDownLatch
                               ){
        int  temp = count.addAndGet(1);
        long  now = System.currentTimeMillis();
        String backTopic = Dict.STREAM_BACK_TOPIC_PREFIX +now+ "_"+sessionId+"_"+temp;
        String sendTopic = Dict.STREAM_SEND_TOPIC_PREFIX +now+"_"+sessionId+"_"+temp;
        context.setTopic(sendTopic);
        context.setActionType(ActionType.MSG_REDIRECT.name());
        context.setQueueType(QueueType.DIRECT.name());
        CreateQueueResult createQueueResult = transferQueueManager.createNewQueue(sessionId,backTopic , true, QueueType.DIRECT);
        if (createQueueResult.getQueue() == null) {
            throw new RemoteRpcException("create queue error");
        }
//        DirectQueue answerQueue = (DirectQueue)createQueueResult.getQueue();
       // consumerManager.createEventDrivenConsumer(backTopic,new SourceGrpcEventHandler(transferQueueManager,respStreamObserver,parser));
        DirectQueue  directQueue =(DirectQueue)createQueueResult.getQueue();
        directQueue.setStreamObserver(respStreamObserver);
        directQueue.setInputParser(new DataParser() {
            @Override
            public Object parse(Object src) {
                try {
                    Osx.PushInbound  inbound =   Osx.PushInbound.parseFrom((byte[]) src);
                    return Proxy.Metadata.parseFrom(inbound.getPayload());
                } catch (InvalidProtocolBufferException e) {
                    e.printStackTrace();
                }
                return null;
            }
        });
        StreamObserver forwardPushReqSO = new StreamObserver<AbstractMessage>() {

            @Override
            public void onNext(AbstractMessage message) {
                try {
                    OsxContext.pushThreadLocalContext(context);
                    context.setMessageFlag(MessageFlag.SENDMSG.name());
                    context.setQueueType(QueueType.DIRECT.name());
                    Osx.Inbound.Builder inboundBuilder = Osx.Inbound.newBuilder();
                    Osx.PushInbound.Builder  pushInboundBuilder = Osx.PushInbound.newBuilder();
                    pushInboundBuilder.setPayload(message.toByteString());
                    pushInboundBuilder.setTopic(sendTopic);
                    inboundBuilder.setPayload(pushInboundBuilder.build().toByteString());
                    Osx.Outbound outbound = (Osx.Outbound)TransferUtil.redirect(context, inboundBuilder.build(), routerInfo, true);
                    TransferUtil.checkResponse(outbound);
                }catch(Exception e){
                    throw ErrorMessageUtil.toGrpcRuntimeException(e);
                }
                finally {
                    OsxContext.popThreadLocalContext();
                }
            }

            @Override
            public void onError(Throwable throwable) {
                try {
                    OsxContext.pushThreadLocalContext(context);
                    context.setMessageFlag(MessageFlag.ERROR.name());
                    ExceptionInfo exceptionInfo = new ExceptionInfo();
                    exceptionInfo.setMessage(throwable.getMessage());
                    String errorData = JsonUtil.object2Json(exceptionInfo);
                    Osx.Inbound.Builder inboundBuilder = TransferUtil.buildInbound(MetaInfo.PROPERTY_FATE_TECH_PROVIDER, srcPartyId, desPartyId, TargetMethod.PRODUCE_MSG.name(),
                                    sendTopic, MessageFlag.ERROR, sessionId, errorData.getBytes(StandardCharsets.UTF_8))
                            .putMetadata(Osx.Metadata.MessageFlag.name(), MessageFlag.ERROR.name());
                    Osx.Outbound outbound = (Osx.Outbound)TransferUtil.redirect(context, inboundBuilder.build(), routerInfo, true);
                    TransferUtil.checkResponse(outbound);
                    countDownLatch.countDown();
                }catch (Exception e){
                    throw ErrorMessageUtil.toGrpcRuntimeException(e);
                }
                finally {
                    OsxContext.popThreadLocalContext();
                }
            }

            @Override
            public void onCompleted() {
                try {
                    OsxContext.pushThreadLocalContext(context);
                    context.setMessageFlag(MessageFlag.COMPELETED.name());
                    Osx.Inbound.Builder inboundBuilder = Osx.Inbound.newBuilder();

                    Osx.PushInbound.Builder  pushInboundBuilder = Osx.PushInbound.newBuilder();
                    pushInboundBuilder.setPayload(ByteString.copyFrom("completed".getBytes(StandardCharsets.UTF_8)));
                    pushInboundBuilder.setTopic(sendTopic);
                    inboundBuilder.setPayload(pushInboundBuilder.build().toByteString());
                    Osx.Outbound outbound =(Osx.Outbound) TransferUtil.redirect(context, inboundBuilder.build(), routerInfo, true);

//                    Osx.Inbound.Builder inboundBuilder = TransferUtil.buildInbound(MetaInfo.PROPERTY_FATE_TECH_PROVIDER, srcPartyId, desPartyId, TargetMethod.PRODUCE_MSG.name(),
//                                    sendTopic, MessageFlag.COMPELETED, sessionId, "completed".getBytes(StandardCharsets.UTF_8))
//                            .putMetadata(Osx.Metadata.MessageFlag.name(), MessageFlag.COMPELETED.name());
//                    Osx.Outbound outbound = TransferUtil.redirect(context, inboundBuilder.build(), routerInfo, true);

//                    TransferUtil.checkResponse(outbound);
                    countDownLatch.countDown();
                } catch (Exception e) {
                    throw ErrorMessageUtil.toGrpcRuntimeException(e);
                }finally {
                    OsxContext.popThreadLocalContext();
                }
            }
        };
        return forwardPushReqSO;

    };


}
