package com.osx.broker.ptp;

import com.google.common.base.Preconditions;
import com.osx.core.constant.Dict;
import com.osx.core.frame.GrpcConnectionFactory;
import com.osx.core.router.RouterInfo;
import com.osx.core.constant.StatusCode;
import com.osx.core.context.Context;
import com.osx.core.exceptions.TransferQueueNotExistException;
import com.osx.core.service.InboundPackage;
import com.osx.broker.ServiceContainer;
import com.osx.broker.consumer.UnaryConsumer;
import com.osx.broker.queue.TransferQueue;
import com.osx.broker.queue.TransferQueueApplyInfo;
import com.osx.broker.util.TransferUtil;
import io.grpc.ManagedChannel;
import io.grpc.stub.StreamObserver;
import org.ppc.ptp.Pcp;
import org.ppc.ptp.PrivateTransferProtocolGrpc;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class PtpConsumeService  extends AbstractPtpServiceAdaptor {

    Logger logger = LoggerFactory.getLogger(PtpConsumeService.class);

    public  PtpConsumeService(){
        this.setServiceName("consume-unary");
    }

    static  final String  CLIENT_CONSUME="client-consume";

    static  final String  DEFUALT_CONSUME= "default-consume";

    static  final String  REDIRECT_CONSUME = "redirect-consume";


    @Override
    protected Pcp.Outbound doService(Context context, InboundPackage<Pcp.Inbound> data) {
        context.setActionType(DEFUALT_CONSUME);
        Pcp.Inbound inbound = data.getBody();
        String  topic  = context.getTopic();


//        FireworkTransfer.ConsumeRequest consumeRequest = data.getBody();
//        String sessionId = consumeRequest.getSessionId();
//        String transferId = consumeRequest.getTransferId();
//        context.setSessionId(sessionId);
//        context.setTransferId(transferId);


        TransferQueue transferQueue = ServiceContainer.transferQueueManager.getQueue(topic);
        if(transferQueue==null){
            TransferQueueApplyInfo transferQueueApplyInfo = ServiceContainer.transferQueueManager.queryGlobleQueue(topic);
            if(transferQueueApplyInfo==null) {
                throw new TransferQueueNotExistException();
            }else{
                String[] args = transferQueueApplyInfo.getInstanceId().split(":");
                String ip = args[0];
                int port = Integer.parseInt(args[1]);
                RouterInfo routerInfo = new RouterInfo();
                routerInfo.setHost(ip);
                routerInfo.setPort(port);
                context.setRouterInfo(routerInfo);
                return  redirect(context,inbound);
            }
        }
        StreamObserver streamObserver=  (StreamObserver) context.getData(Dict.RESPONSE_STREAM_OBSERVER);
        Long  offset = context.getRequestMsgIndex();
        Preconditions.checkArgument(offset!=null);
        if(offset>0) {
            context.setActionType(CLIENT_CONSUME);
        }
        UnaryConsumer consumer = ServiceContainer.consumerManager.getOrCreateUnaryConsumer(topic);
        TransferQueue.TransferQueueConsumeResult transferQueueConsumeResult = consumer.consume(context, offset);
        context.setReturnCode(transferQueueConsumeResult.getCode());
        if (transferQueueConsumeResult.getCode().equals(StatusCode.CONSUME_NO_MESSAGE)) {
            /*
             *   由其他扫描线程应答
             */
            if(offset<0) {
                UnaryConsumer.LongPullingHold longPullingHold = new UnaryConsumer.LongPullingHold();
                longPullingHold.setNeedOffset(offset);
                longPullingHold.setStreamObserver(streamObserver);
                longPullingHold.setContext(context.subContext());
                consumer.addLongPullingQueue(longPullingHold);
                return null;
            }
        }
        Pcp.Outbound consumeResponse = TransferUtil.buildResponse(transferQueueConsumeResult.getCode(), "", transferQueueConsumeResult);
        return  consumeResponse;

    }

    private Pcp.Outbound redirect(Context  context , Pcp.Inbound inbound ){
        ManagedChannel managedChannel = GrpcConnectionFactory.createManagedChannel(context.getRouterInfo());
        context.setActionType(REDIRECT_CONSUME);
        PrivateTransferProtocolGrpc.PrivateTransferProtocolBlockingStub stub = PrivateTransferProtocolGrpc.newBlockingStub(managedChannel);
        return stub.invoke(inbound);
    }



}
