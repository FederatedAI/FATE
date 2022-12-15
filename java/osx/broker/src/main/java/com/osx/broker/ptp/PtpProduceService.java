package com.osx.broker.ptp;

import com.osx.core.config.MetaInfo;
import com.osx.core.constant.DeployMode;
import com.osx.core.constant.StatusCode;
import com.osx.core.context.Context;
import com.osx.core.exceptions.InvalidRedirectInfoException;
import com.osx.core.exceptions.ParameterException;
import com.osx.core.exceptions.ProduceMsgExcption;
import com.osx.core.exceptions.PutMessageException;
import com.osx.core.router.RouterInfo;
import com.osx.core.service.InboundPackage;
import com.osx.broker.ServiceContainer;
import com.osx.broker.constants.Direction;
import com.osx.broker.grpc.MessageFlag;
import com.osx.broker.message.MessageDecoder;
import com.osx.broker.message.MessageExtBrokerInner;
import com.osx.broker.queue.CreateQueueResult;
import com.osx.broker.queue.PutMessageResult;
import com.osx.broker.queue.PutMessageStatus;
import com.osx.broker.queue.TransferQueue;
import com.osx.broker.util.ResourceUtil;
import org.apache.commons.lang3.StringUtils;

import org.ppc.ptp.Pcp;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static com.osx.broker.util.TransferUtil.redirect;

public class PtpProduceService extends AbstractPtpServiceAdaptor {

    Logger logger = LoggerFactory.getLogger(PtpProduceService.class);

    @Override
    protected Pcp.Outbound doService(Context context, InboundPackage<Pcp.Inbound> data) {

        String topic = context.getTopic();
        boolean isDst=false;
        RouterInfo   routerInfo = context.getRouterInfo();
        String srcPartyId =  context.getSrcPartyId();
        String  sessionId = context.getSessionId();
        Pcp.Inbound produceRequest = data.getBody();
        if(MetaInfo.PROPERTY_SELF_PARTY.contains(context.getDesPartyId())){
            isDst= true;
        }
        if(!isDst) {
            /**
             * 向外转发
             */
                return redirect(context, produceRequest, routerInfo, false);
        }
        else{
            /**
             * 本地处理
             */
            if(StringUtils.isEmpty(topic)){
                throw  new ParameterException(StatusCode.PARAM_ERROR,"topic is null");
            }
            if(StringUtils.isEmpty(sessionId)){
                throw  new ParameterException(StatusCode.PARAM_ERROR,"sessionId is null");
            }
            context.setActionType("download");
            CreateQueueResult createQueueResult = ServiceContainer.transferQueueManager.createNewQueue(topic,sessionId,false);
            if(createQueueResult==null){
                throw new RuntimeException("transfer queue is null");
            }
            String resource = ResourceUtil.buildResource(srcPartyId+"-"+MetaInfo.PROPERTY_SELF_PARTY, Direction.UP);

            if(createQueueResult==null){
                throw new RuntimeException("transfer queue is null");
            }

            //String resource = ResourceUtil.buildResource(context.getSrcPartyId()+"-"+MetaInfo.PROPERTY_SELF_PARTY, Direction.UP);
            // ServiceContainer.tokenApplyService.applyToken(context,resource,dataSize);
            //ServiceContainer.flowCounterManager.pass(resource,dataSize);
            TransferQueue  transferQueue = createQueueResult.getTransferQueue();
            if(transferQueue!=null) {
                //MessageExtBrokerInner messageExtBrokerInner = new MessageExtBrokerInner();
                byte[] msgBytes = produceRequest.getPayload().toByteArray();
                //context.(msgBytes.length);
                //messageExtBrokerInner.setBody(msgBytes);

                /**
                 * 这里写成blank 是因为topic长度太长 ，如果调整字节后是可以在这里设置的
                 */
                // messageExtBrokerInner.setTopic("blank");
                MessageExtBrokerInner messageExtBrokerInner  =  MessageDecoder.buildMessageExtBrokerInner(topic,msgBytes,0, MessageFlag.MSG,context.getSrcPartyId(),
                        context.getDesPartyId());
                PutMessageResult putMessageResult = transferQueue.putMessage(messageExtBrokerInner);
                if(putMessageResult.getPutMessageStatus()!= PutMessageStatus.PUT_OK){
                    throw  new PutMessageException("put status "+putMessageResult.getPutMessageStatus());
                }
                long  logicOffset = putMessageResult.getMsgLogicOffset();
                //context.setDataSize(logicOffset);
//            FireworkTransfer.ProduceResponse.Builder produceResponseBuilder = FireworkTransfer.ProduceResponse.newBuilder();
//            produceResponseBuilder.setCode(StatusCode.SUCCESS);
//            produceResponseBuilder.setMsg("SUCCESS");
                Pcp.Outbound.Builder   outBoundBuilder =  Pcp.Outbound.newBuilder();
                outBoundBuilder.setCode(StatusCode.SUCCESS);
                outBoundBuilder.setMessage("SUCCESS");
                return outBoundBuilder.build();
            }else{
                /**
                 * 集群内转发
                 */

                if(MetaInfo.PROPERTY_DEPLOY_MODE.equals(DeployMode.cluster.name())) {
                    RouterInfo redirectRouterInfo = new RouterInfo();
                    String redirectIp = createQueueResult.getRedirectIp();
                    int redirectPort = createQueueResult.getPort();
                    if(StringUtils.isEmpty(redirectIp)||redirectPort==0){
                        logger.error("invalid redirect info {}:{}",redirectIp,redirectPort);
                        throw new InvalidRedirectInfoException();
                    }
                    redirectRouterInfo.setHost(redirectIp);
                    redirectRouterInfo.setPort(redirectPort);
                    context.setRouterInfo(redirectRouterInfo);
                    context.setActionType("inner-redirect");
                    return redirect(context ,produceRequest, redirectRouterInfo,true);
                }else{
                    logger.error("create topic {} error",topic);
                    throw new ProduceMsgExcption();
                }
            }
        }
    }




}
