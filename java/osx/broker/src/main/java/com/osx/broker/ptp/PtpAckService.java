package com.osx.broker.ptp;

import com.osx.core.constant.Dict;
import com.osx.core.config.MetaInfo;
import com.osx.core.router.RouterInfo;
import com.osx.core.constant.StatusCode;
import com.osx.core.context.Context;
import com.osx.core.exceptions.ConsumerNotExistException;
import com.osx.core.exceptions.InvalidRedirectInfoException;
import com.osx.core.exceptions.TransferQueueNotExistException;
import com.osx.core.service.InboundPackage;
import com.osx.broker.ServiceContainer;
import com.osx.broker.consumer.UnaryConsumer;
import com.osx.broker.queue.TransferQueue;
import com.osx.broker.queue.TransferQueueApplyInfo;
import org.apache.commons.lang3.StringUtils;
import org.ppc.ptp.Pcp;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static com.osx.broker.util.TransferUtil.redirect;

public class PtpAckService extends AbstractPtpServiceAdaptor{
    Logger logger = LoggerFactory.getLogger(PtpAckService.class);

    @Override
    protected Pcp.Outbound doService(Context context, InboundPackage<Pcp.Inbound> data) {
        context.setActionType("local-ack");
        Pcp.Inbound inbound = data.getBody();
        Pcp.Outbound.Builder outboundBuilder = Pcp.Outbound.newBuilder();
        String sessionId=context.getSessionId();
        String topic = context.getTopic();
        Long offset = context.getRequestMsgIndex();
        TransferQueue transferQueue = ServiceContainer.transferQueueManager.getQueue(topic);
        /**
         * 若本地queue不存在，则检查是否在集群中其他节点
         */
        if(transferQueue==null) {
            if (MetaInfo.isCluster()) {
                TransferQueueApplyInfo transferQueueApplyInfo =  ServiceContainer.transferQueueManager.queryGlobleQueue(topic);
                if(transferQueueApplyInfo==null){
                    throw new TransferQueueNotExistException();
                }else{
                    context.setActionType("redirect-ack");
                    String[] ipport =transferQueueApplyInfo.getInstanceId().split(":");

                    RouterInfo redirectRouterInfo = new RouterInfo();
                    String redirectIp = ipport[0];
                    int redirectPort = Integer.parseInt(ipport[1]);
                    if(StringUtils.isEmpty(redirectIp)||redirectPort==0){
                        logger.error("invalid redirect info {}:{}",redirectIp,redirectPort);
                        throw new InvalidRedirectInfoException();
                    }
                    redirectRouterInfo.setHost(redirectIp);
                    redirectRouterInfo.setPort(redirectPort);
                    //context.setRouterInfo(redirectRouterInfo);
                    context.setActionType("inner-redirect");
                    return  redirect(context,inbound,redirectRouterInfo,false);
                }
            } else {
                throw new TransferQueueNotExistException();
            }
        }
        UnaryConsumer unaryConsumer = ServiceContainer.consumerManager.getUnaryConsumer(topic);
        if(unaryConsumer!=null){
            long  currentMsgIndex = unaryConsumer.ack(offset);
            context.setCurrentMsgIndex(currentMsgIndex);
            outboundBuilder.setCode(StatusCode.SUCCESS);
            outboundBuilder.setMessage(Dict.SUCCESS);
            return  outboundBuilder.build();
        }else{
            throw  new ConsumerNotExistException("consumer is not exist");
        }

    }


}
