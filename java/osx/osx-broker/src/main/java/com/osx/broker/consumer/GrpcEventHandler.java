package com.osx.broker.consumer;


import com.osx.api.context.Context;
import com.osx.api.router.RouterInfo;
import com.osx.broker.ServiceContainer;
import com.osx.broker.constants.MessageFlag;
import com.osx.broker.message.MessageExt;
import com.osx.broker.queue.TransferQueue;
import com.osx.broker.util.TransferUtil;
import com.osx.core.config.MetaInfo;
import com.osx.core.constant.Dict;
import com.osx.core.constant.StatusCode;
import com.osx.core.constant.TransferStatus;
import com.osx.core.context.FateContext;
import com.osx.core.exceptions.ExceptionInfo;
import com.osx.core.frame.GrpcConnectionFactory;
import com.osx.core.ptp.TargetMethod;
import io.grpc.ManagedChannel;
import org.ppc.ptp.Osx;
import org.ppc.ptp.PrivateTransferProtocolGrpc;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.charset.StandardCharsets;

public abstract class GrpcEventHandler  {

    Logger logger = LoggerFactory.getLogger(GrpcEventHandler.class);
    public  GrpcEventHandler(String provider){
        this.provider = provider;
    }
    protected TransferStatus transferStatus = TransferStatus.INIT;
    protected String provider;
    protected String srcPartyId;
    protected String desPartyId;
    protected String sessionId;
    protected String srcComponent;
    protected String desComponent;
    protected String topic;
    protected String backTopic;
    protected RouterInfo backRouterInfo;
    protected FateContext context;

    public  void  sendBackException(ExceptionInfo e){
        if(transferStatus==TransferStatus.TRANSFERING) {
            Osx.Inbound.Builder inboundBuilder = TransferUtil.buildInbound(provider,desPartyId, srcPartyId,
                    TargetMethod.PRODUCE_MSG.name(), backTopic, MessageFlag.COMPELETED, sessionId,e.toString().getBytes(StandardCharsets.UTF_8) );
            TransferUtil.redirect(context,inboundBuilder.build(),backRouterInfo,true);

        }else{
            logger.error("!!!!!!!!!transferStatus is {}",transferStatus);
        }
    };

    public void  sendBackCompleted(){
        if(transferStatus== TransferStatus.TRANSFERING) {
            Osx.Inbound.Builder inboundBuilder = TransferUtil.buildInbound(provider,desPartyId, srcPartyId,
                    TargetMethod.PRODUCE_MSG.name(), backTopic, MessageFlag.COMPELETED, sessionId, "completed".getBytes(StandardCharsets.UTF_8));
            TransferUtil.redirect(context,inboundBuilder.build(),backRouterInfo,true);
        }else{
            logger.error("!!!!!!!!!transferStatus is {}",transferStatus);
        }
    }

    public void  sendBackMsg(byte[]  data){
        if(transferStatus== TransferStatus.TRANSFERING) {
            Osx.Inbound.Builder inboundBuilder = TransferUtil.buildInbound(provider,desPartyId, srcPartyId,
                    TargetMethod.PRODUCE_MSG.name(), backTopic, MessageFlag.SENDMSG, sessionId, data);
            TransferUtil.redirect(context,inboundBuilder.build(),backRouterInfo,true);
        }else{
            logger.error("!!!!!!!!!transferStatus is {}",transferStatus);
        }
    }

    protected void init(MessageExt message){

        if(transferStatus==TransferStatus.INIT){
            try {

//                messageEvent.setDesComponent(message.getProperty(Dict.DES_COMPONENT));
//                messageEvent.setSrcComponent(message.getProperty(Dict.SOURCE_COMPONENT));
//                messageEvent.setSrcPartyId(message.getSrcPartyId());
//                messageEvent.setDesPartyId(message.getDesPartyId());
//                messageEvent.setSessionId(message.getProperty(Dict.SESSION_ID));
                context =  new FateContext();
                topic = message.getTopic();
                desComponent = message.getProperty(Dict.DES_COMPONENT);
                srcComponent = message.getProperty(Dict.SOURCE_COMPONENT);
                srcPartyId = message.getSrcPartyId();
                desPartyId = message.getDesPartyId();
                sessionId = message.getProperty(Dict.SESSION_ID);
                if (topic.startsWith(Dict.STREAM_SEND_TOPIC_PREFIX)) {
                    backTopic = topic.replaceAll(Dict.STREAM_SEND_TOPIC_PREFIX, Dict.STREAM_BACK_TOPIC_PREFIX);
                } else if (topic.startsWith(Dict.STREAM_BACK_TOPIC_PREFIX)) {
                    backTopic = topic.replaceAll(Dict.STREAM_BACK_TOPIC_PREFIX, Dict.STREAM_SEND_TOPIC_PREFIX);
                }
                backRouterInfo = ServiceContainer.routerRegister.getRouterService(MetaInfo.PROPERTY_FATE_TECH_PROVIDER).route(desPartyId,"",srcPartyId,"");
                handleInit(message);
                transferStatus = TransferStatus.TRANSFERING;
            }catch(Throwable e){
                logger.error("grpc event handler init error",e);
                transferStatus = TransferStatus.ERROR;
            }
        }


    }



    public void onEvent(MessageExt messageExt) throws Exception {

     //   String  topic =  event.getTopic();

//        messageEvent.setDesComponent(message.getProperty(Dict.DES_COMPONENT));
//        messageEvent.setSrcComponent(message.getProperty(Dict.SOURCE_COMPONENT));
//        messageEvent.setSrcPartyId(message.getSrcPartyId());
//        messageEvent.setDesPartyId(message.getDesPartyId());
//        messageEvent.setSessionId(message.getProperty(Dict.SESSION_ID));

       // logger.info("======event {}",event);
        init(messageExt);
        if(transferStatus==TransferStatus.TRANSFERING) {
//            EventDrivenConsumer consumer = ServiceContainer.consumerManager.getEventDrivenConsumer(topic);
//            TransferQueue.TransferQueueConsumeResult transferQueueConsumeResult = consumer.consume(new FateContext(), -1);
//
//            if (transferQueueConsumeResult.getCode().equals(StatusCode.SUCCESS)) {
//                long index = transferQueueConsumeResult.getRequestIndex();
//                //ack 的位置需要调整
//                consumer.ack(index);
//                MessageExt messageExt = transferQueueConsumeResult.getMessage();
//
                int flag = messageExt.getFlag();
              //  logger.info("message flag {}", flag);
                switch (flag) {
                    //msg
                    case 0:
                        handleMessage(messageExt);
                        break;
                    //error
                    case 1:
                        handleError(messageExt);
                        break;
                    //completed
                    case 2:
                        handleComplete(messageExt);
                        break;
                    default:
                        ;
                }
//            } else {
//             //   logger.warn("consume error {}", transferQueueConsumeResult);
//            }
        }
    }

    protected abstract void  handleMessage(MessageExt message);
    protected abstract void  handleError(MessageExt message);
    protected abstract void  handleComplete(MessageExt message);
    protected abstract  void  handleInit(MessageExt message);

}
