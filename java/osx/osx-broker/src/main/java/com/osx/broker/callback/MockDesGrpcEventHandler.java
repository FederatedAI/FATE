//package com.osx.broker.callback;
//
//import com.google.protobuf.ByteString;
//import com.google.protobuf.InvalidProtocolBufferException;
//import com.osx.broker.ServiceContainer;
//import com.osx.broker.constants.MessageFlag;
//import com.osx.broker.consumer.GrpcEventHandler;
//import com.osx.broker.consumer.MessageEvent;
//import com.osx.broker.message.MessageExt;
//import com.osx.broker.util.TransferUtil;
//import com.osx.core.constant.Dict;
//import com.osx.core.constant.TransferStatus;
//import com.osx.core.frame.GrpcConnectionFactory;
//import com.osx.core.ptp.TargetMethod;
//import com.osx.core.router.RouterInfo;
//import com.webank.ai.eggroll.api.networking.proxy.DataTransferServiceGrpc;
//import io.grpc.ManagedChannel;
//import org.ppc.ptp.Osx;
//import org.ppc.ptp.PrivateTransferProtocolGrpc;
//import org.slf4j.Logger;
//import org.slf4j.LoggerFactory;
//
//import java.nio.charset.StandardCharsets;
//
//public class MockDesGrpcEventHandler extends GrpcEventHandler {
//
//
//
//    Logger logger = LoggerFactory.getLogger(MockDesGrpcEventHandler.class);
//
//    PrivateTransferProtocolGrpc.PrivateTransferProtocolBlockingStub blockingStub;
//    @Override
//    protected void handleMessage(MessageExt message) {
//
//        String topic  = message.getTopic();
//        String  srcPartyId = message.getSrcPartyId();
//        String  desPartyId = message .getDesPartyId();
//        try {
//            Osx.Inbound  inbound =    Osx.Inbound.parseFrom(message.getBody());
//            logger.info("receive message topic {} srcPartyId {} desPartyId {} msg {}",topic,srcPartyId,desPartyId,new String(inbound.getPayload().toByteArray()));
//        } catch (InvalidProtocolBufferException e) {
//            e.printStackTrace();
//        }
//
//    }
//
//    @Override
//    protected void handleError(MessageExt message) {
//        logger.info("handle error : {}",new String(message.getBody()));
//    }
//
//    @Override
//    protected void handleComplete(MessageExt message) {
//        logger.info("receive complete");
//
//    }
//
//    @Override
//    protected void handleInit(MessageEvent event) {
//
//        logger.info("init================= {} {} {} {} {}",topic, backTopic,srcPartyId,desPartyId,sessionId);
//        new Thread(new Runnable() {
//            @Override
//            public void run() {
//                for(int i=0;i<10;i++){
//
//                    Osx.Outbound outBound = Osx.Outbound.newBuilder().setPayload(ByteString.copyFrom("my name is god".getBytes(StandardCharsets.UTF_8))).build();
//                    sendBackMsg(outBound.toByteArray());
//                    if(i==9){
//                        sendBackCompleted();
//                    }
//                }
//            }
//        }).start();
//    }
//
//
//}
