//package  org.fedai.osx.broker.callback;
//
//import com.google.protobuf.ByteString;
//import com.google.protobuf.InvalidProtocolBufferException;
//import  org.fedai.osx.broker.ServiceContainer;
//import  org.fedai.osx.broker.constants.MessageFlag;
//import  org.fedai.osx.broker.consumer.GrpcEventHandler;
//import  org.fedai.osx.broker.consumer.MessageEvent;
//import  org.fedai.osx.broker.message.MessageExt;
//import  org.fedai.osx.broker.util.TransferUtil;
//import  org.fedai.osx.core.constant.Dict;
//import  org.fedai.osx.core.constant.TransferStatus;
//import  org.fedai.osx.core.frame.GrpcConnectionFactory;
//import  org.fedai.osx.core.ptp.TargetMethod;
//import  org.fedai.osx.core.router.RouterInfo;
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
