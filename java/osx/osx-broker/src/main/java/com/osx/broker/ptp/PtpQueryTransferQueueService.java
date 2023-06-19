/*
 * Copyright 2019 The FATE Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.osx.broker.ptp;


import com.osx.broker.ServiceContainer;
import com.osx.broker.queue.CreateQueueResult;
import com.osx.broker.queue.TransferQueue;
import com.osx.broker.queue.TransferQueueApplyInfo;
import com.osx.core.config.MetaInfo;
import com.osx.core.constant.ActionType;
import com.osx.core.constant.StatusCode;
import com.osx.core.context.FateContext;
import com.osx.core.service.InboundPackage;
import com.osx.core.utils.NetUtils;
import org.ppc.ptp.Osx;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Map;

public class PtpQueryTransferQueueService extends AbstractPtpServiceAdaptor {
    Logger logger = LoggerFactory.getLogger(PtpQueryTransferQueueService.class);

    public PtpQueryTransferQueueService() {
        this.setServiceName("query");
    }

    @Override
    protected Osx.Outbound doService(FateContext context, InboundPackage<Osx.Inbound> data) {

        Osx.Inbound request = data.getBody();
        Osx.Outbound.Builder outboundBuilder = Osx.Outbound.newBuilder();
        Map<String, String> metaDataMap = request.getMetadataMap();
        Osx.TopicInfo topicInfo;
//        String version = metaDataMap.get(Pcp.Header.Version.name());
//        String techProviderCode = metaDataMap.get(Pcp.Header.TechProviderCode.name());
//        String traceId = metaDataMap.get(Pcp.Header.TraceID.name());
//        String token = metaDataMap.get(Pcp.Header.Token.name());
//        String sourceNodeId = metaDataMap.get(Pcp.Header.SourceNodeID.name());
//        String targetNodeId = metaDataMap.get(Pcp.Header.TargetNodeID.name());
//        String sourceInstId = metaDataMap.get(Pcp.Header.SourceInstID.name());
//        String targetInstId = metaDataMap.get(Pcp.Header.TargetInstID.name());
        String sessionId = metaDataMap.get(Osx.Header.SessionID.name());
//        String targetMethod = metaDataMap.get(Pcp.Metadata.TargetMethod.name());
//        String targetComponentName = metaDataMap.get(Pcp.Metadata.TargetComponentName.name());
//        String sourceComponentName = metaDataMap.get(Pcp.Metadata.SourceComponentName.name());
//        String sourcePartyId= sourceInstId+"."+sourceNodeId;
//        String targetPartyId = targetNodeId+"."+targetNodeId;
        String topic = metaDataMap.get(Osx.Metadata.MessageTopic.name());
        //       String offsetString = metaDataMap.get(Pcp.Metadata.MessageOffSet.name());
        context.setActionType(ActionType.QUERY_TOPIC.getAlias());
//        FireworkTransfer.QueryTransferQueueInfoRequest queryTransferQueueInfoRequest = data.getBody();
//        String  transferId = queryTransferQueueInfoRequest.getTransferId();
//        String  sessionId = queryTransferQueueInfoRequest.getSessionId();
        context.setSessionId(sessionId);
        context.setTopic(topic);
//        logger.info("receive query request {}",transferId);
        //Preconditions.checkArgument(StringUtils.isNotEmpty(transferId));
        TransferQueue transferQueue = ServiceContainer.transferQueueManager.getQueue(topic);
        //FireworkTransfer.QueryTransferQueueInfoResponse response;
        Osx.TopicInfo.Builder topicInfoBuilder = Osx.TopicInfo.newBuilder();
        if (transferQueue != null) {
            topicInfo = topicInfoBuilder.setTopic(transferQueue.getTransferId()).
                    setCreateTimestamp(transferQueue.getCreateTimestamp())
                    .setIp(NetUtils.getLocalHost())
                    .setPort(MetaInfo.PROPERTY_GRPC_PORT).build();
        } else {
            /**
             * 全局topic信息
             */
            TransferQueueApplyInfo transferQueueApplyInfo = ServiceContainer.transferQueueManager.queryGlobleQueue(topic);
            if (transferQueueApplyInfo != null) {
                String instanceId = transferQueueApplyInfo.getInstanceId();
                String[] instanceElements = instanceId.split(":");
                topicInfoBuilder.setTopic(transferQueueApplyInfo.getTransferId()).
                        setCreateTimestamp(transferQueueApplyInfo.getApplyTimestamp())
                        .setIp(instanceElements[0])
                        .setPort(Integer.parseInt(instanceElements[1]));
                topicInfo = topicInfoBuilder.build();
            } else {
                /**
                 * 由查询创建队列
                 */
                CreateQueueResult createQueueResult = ServiceContainer.transferQueueManager.createNewQueue(topic, sessionId, false);
                topicInfo = topicInfoBuilder
                        .setTopic(topic)
                        .setCreateTimestamp(System.currentTimeMillis())
                        .setIp(createQueueResult.getRedirectIp())
                        .setPort(createQueueResult.getPort())
                        .build();
            }
        }
        outboundBuilder.setCode(StatusCode.SUCCESS);
        outboundBuilder.setMessage("SUCCESS");
        outboundBuilder.setPayload(topicInfo.toByteString());
        return outboundBuilder.build();
    }


//    @Override
//    protected FireworkTransfer.QueryTransferQueueInfoResponse doService(Context context, InboundPackage<FireworkTransfer.QueryTransferQueueInfoRequest> data, OutboundPackage<FireworkTransfer.QueryTransferQueueInfoResponse> outboundPackage) {
//        context.setActionType("query");
//        FireworkTransfer.QueryTransferQueueInfoRequest queryTransferQueueInfoRequest = data.getBody();
//        String  transferId = queryTransferQueueInfoRequest.getTransferId();
//        String  sessionId = queryTransferQueueInfoRequest.getSessionId();
//        context.setSessionId(sessionId);
//        context.setTransferId(transferId);
////        logger.info("receive query request {}",transferId);
//        //Preconditions.checkArgument(StringUtils.isNotEmpty(transferId));
//        TransferQueue transferQueue = this.transferQueueManager.getQueue(transferId);
//        FireworkTransfer.QueryTransferQueueInfoResponse response;
//        if(transferQueue!=null){
//            response = FireworkTransfer.QueryTransferQueueInfoResponse.newBuilder().
//                    addTransferQueueInfo(FireworkTransfer.TransferQueueInfo.newBuilder()
//                            .setTransferId(transferQueue.getTransferId())
//                            .setCreateTimestamp(transferQueue.getCreateTimestamp())
//                            .setIp(NetUtils.getLocalHost())
//                            .setPort(MetaInfo.PROPERTY_PORT)
//                            .build()).setCode(0)
//                    .build();
//        }else {
//            TransferQueueApplyInfo transferQueueApplyInfo = this.transferQueueManager.queryGlobleQueue(transferId);
//            if(transferQueueApplyInfo!=null){
//
//                String instanceId = transferQueueApplyInfo.getInstanceId();
//                String[] instanceElements = instanceId.split(":");
//                response = FireworkTransfer.QueryTransferQueueInfoResponse.newBuilder().
//                        addTransferQueueInfo(FireworkTransfer.TransferQueueInfo.newBuilder()
//                                .setTransferId(transferQueueApplyInfo.getTransferId())
//                                .setCreateTimestamp(transferQueueApplyInfo.getApplyTimestamp())
//                                .setIp(instanceElements[0])
//                                .setPort(Integer.parseInt(instanceElements[1]))
//                                .build()).setCode(0)
//                        .build();
//
//            }else{
//                /**
//                 * 由查询创建队列
//                 */
//                CreateQueueResult createQueueResult = this.transferQueueManager.createNewQueue(transferId,sessionId,false);
//                response = FireworkTransfer.QueryTransferQueueInfoResponse.newBuilder().
//                        addTransferQueueInfo(FireworkTransfer.TransferQueueInfo.newBuilder()
//                                .setTransferId(transferId)
//                                .setCreateTimestamp(System.currentTimeMillis())
//                                .setIp(createQueueResult.getRedirectIp())
//                                .setPort(createQueueResult.getPort())
//                                .build()).setCode(0)
//                        .build();
//
////                response = FireworkTransfer.QueryTransferQueueInfoResponse.newBuilder().set
////                        .setCode(StatusCode.TRANSFER_QUEUE_NOT_FIND).build();
//               // context.setReturnCode(StatusCode.TRANSFER_QUEUE_NOT_FIND);
//            }
//        }
//        return response;
//    }


}
