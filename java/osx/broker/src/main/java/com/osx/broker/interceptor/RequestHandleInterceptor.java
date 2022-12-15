package com.osx.broker.interceptor;
import com.osx.core.router.RouterInfo;
import com.osx.core.context.Context;
import com.osx.core.service.InboundPackage;
import com.osx.core.service.Interceptor;
import com.osx.core.exceptions.NoRouterInfoException;
import com.osx.broker.grpc.PushRequestDataWrap;
import com.osx.broker.router.FateRouterService;
import org.apache.commons.lang3.StringUtils;
import org.ppc.ptp.Pcp;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import com.webank.ai.eggroll.api.networking.proxy.Proxy;

import java.util.Map;

public class RequestHandleInterceptor implements Interceptor {

    Logger logger = LoggerFactory.getLogger(RequestHandleInterceptor.class);

    public  RequestHandleInterceptor(    FateRouterService  fateRouterService){
        this.fateRouterService = fateRouterService;
    }

    FateRouterService  fateRouterService;

    public void doPreProcess(Context context, InboundPackage inboundPackage) throws Exception {
        Object  body = inboundPackage.getBody();

        if(body instanceof PushRequestDataWrap){

            PushRequestDataWrap  pushRequestDataWrap = (PushRequestDataWrap) body;
            Proxy.Packet  packet = pushRequestDataWrap.getPacket();
            context.putData("pushStreamObserver",pushRequestDataWrap.getStreamObserver());
            RouterInfo routerInfo =  fateRouterService.route(packet);
            if(packet!=null&&packet.getHeader()!=null&&packet.getHeader().getTask()!=null) {
                context.setTopic(packet.getHeader().getTask().getTaskId());
            }
            //Preconditions.checkArgument(routerInfo!=null);
            context.setDesPartyId(routerInfo.getDesPartyId());
            context.setSrcPartyId(routerInfo.getSourcePartyId());
            context.setRouterInfo(routerInfo);
        }
        if(body instanceof Pcp.Inbound){
                Pcp.Inbound  request  = (Pcp.Inbound)body;
                Map<String,String> metaDataMap = request.getMetadataMap();

                String version = metaDataMap.get(Pcp.Header.Version.name());
                String techProviderCode = metaDataMap.get(Pcp.Header.TechProviderCode.name());
                String traceId = metaDataMap.get(Pcp.Header.TraceID.name());
                String token = metaDataMap.get(Pcp.Header.Token.name());
                String sourceNodeId = metaDataMap.get(Pcp.Header.SourceNodeID.name());
                String targetNodeId = metaDataMap.get(Pcp.Header.TargetNodeID.name());
                String sourceInstId = metaDataMap.get(Pcp.Header.SourceInstID.name());
                String targetInstId = metaDataMap.get(Pcp.Header.TargetInstID.name());
                String sessionId = metaDataMap.get(Pcp.Header.SessionID.name());
                String targetMethod = metaDataMap.get(Pcp.Metadata.TargetMethod.name());
                String targetComponentName = metaDataMap.get(Pcp.Metadata.TargetComponentName.name());
                String sourceComponentName = metaDataMap.get(Pcp.Metadata.SourceComponentName.name());
                String sourcePartyId= StringUtils.isEmpty(sourceInstId)?sourceNodeId:sourceInstId+"."+sourceNodeId;
                String targetPartyId =StringUtils.isEmpty(targetInstId)? targetNodeId:targetInstId+"."+targetNodeId;
                String topic =  metaDataMap.get(Pcp.Metadata.MessageTopic.name());
                String offsetString = metaDataMap.get(Pcp.Metadata.MessageOffSet.name());
                RouterInfo routerInfo = fateRouterService.route(sourcePartyId,sourceComponentName,targetPartyId,targetComponentName);
                Long  offset = StringUtils.isNotEmpty(offsetString)?Long.parseLong(offsetString):null;

            context.setDesPartyId(targetPartyId);
            context.setSrcPartyId(sourcePartyId);
            context.setRouterInfo(routerInfo);
            context.setTopic(topic);
            context.setRequestMsgIndex(offset);
            context.setSessionId(sessionId);

            logger.info("=========== sessionId {}pppppppppppp{}",context.getSessionId(),sessionId);
            logger.info("metaDataMap {}",metaDataMap);
            return ;
        }


        /**
         * 旧版本
         */
        if(body instanceof  Proxy.Packet){
            Proxy.Packet  packet = (Proxy.Packet) body;
            RouterInfo  routerInfo =  fateRouterService.route((Proxy.Packet)body);
            if(routerInfo==null){
                throw new NoRouterInfoException("no router info");
            }
            if(packet!=null&&packet.getHeader()!=null&&packet.getHeader().getTask()!=null) {
                context.setTopic(packet.getHeader().getTask().getTaskId());
            }
            context.setDesPartyId(routerInfo.getDesPartyId());
            context.setSrcPartyId(routerInfo.getSourcePartyId());
            context.setRouterInfo(routerInfo);
        }else{
            throw  new RuntimeException();
        }
    }

}
