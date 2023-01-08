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
package com.osx.broker.interceptor;

import com.google.protobuf.InvalidProtocolBufferException;
import com.osx.broker.grpc.PushRequestDataWrap;
import com.osx.broker.router.FateRouterService;
import com.osx.core.context.Context;
import com.osx.core.exceptions.NoRouterInfoException;
import com.osx.core.exceptions.ParameterException;
import com.osx.core.router.RouterInfo;
import com.osx.core.service.InboundPackage;
import com.osx.core.service.Interceptor;
import com.webank.ai.eggroll.api.networking.proxy.Proxy;
import com.webank.eggroll.core.transfer.Transfer;
import org.apache.commons.lang3.StringUtils;
import org.ppc.ptp.Osx;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Map;

public class RequestHandleInterceptor implements Interceptor {
    Logger logger = LoggerFactory.getLogger(RequestHandleInterceptor.class);

    public void doPreProcess(Context context, InboundPackage inboundPackage) throws Exception {
        Object body = inboundPackage.getBody();

     if (body instanceof Osx.Inbound) {
            Osx.Inbound request = (Osx.Inbound) body;
            Map<String, String> metaDataMap = request.getMetadataMap();
            String version = metaDataMap.get(Osx.Header.Version.name());
            String techProviderCode = metaDataMap.get(Osx.Header.TechProviderCode.name());
            String traceId = metaDataMap.get(Osx.Header.TraceID.name());
            String token = metaDataMap.get(Osx.Header.Token.name());
            String sourceNodeId = metaDataMap.get(Osx.Header.SourceNodeID.name());
            String targetNodeId = metaDataMap.get(Osx.Header.TargetNodeID.name());
            String sourceInstId = metaDataMap.get(Osx.Header.SourceInstID.name());
            String targetInstId = metaDataMap.get(Osx.Header.TargetInstID.name());
            String sessionId = metaDataMap.get(Osx.Header.SessionID.name());
            String targetMethod = metaDataMap.get(Osx.Metadata.TargetMethod.name());
            String targetComponentName = metaDataMap.get(Osx.Metadata.TargetComponentName.name());
            String sourceComponentName = metaDataMap.get(Osx.Metadata.SourceComponentName.name());
            String sourcePartyId = StringUtils.isEmpty(sourceInstId) ? sourceNodeId : sourceInstId + "." + sourceNodeId;
            String targetPartyId = StringUtils.isEmpty(targetInstId) ? targetNodeId : targetInstId + "." + targetNodeId;
            String topic = metaDataMap.get(Osx.Metadata.MessageTopic.name());
            String offsetString = metaDataMap.get(Osx.Metadata.MessageOffSet.name());
            Long offset = StringUtils.isNotEmpty(offsetString) ? Long.parseLong(offsetString) : null;
            context.setDesPartyId(targetPartyId);
            context.setSrcPartyId(sourcePartyId);
            context.setTopic(topic);
            context.setRequestMsgIndex(offset);
            context.setSessionId(sessionId);
            context.setDesComponent(targetComponentName);
            context.setSrcComponent(sourceComponentName);
            return;
        }
        else if (body instanceof PushRequestDataWrap) {
            PushRequestDataWrap pushRequestDataWrap = (PushRequestDataWrap) body;
            Proxy.Packet packet = pushRequestDataWrap.getPacket();
            handleProxyPacket(context ,packet);
            return ;
        }else if (body instanceof Proxy.Packet) {
         handleProxyPacket(context ,(Proxy.Packet) body);
     } else {
         throw new ParameterException("invalid inbound type");
     }

    }

    private   void  handleProxyPacket(Context context ,Proxy.Packet packet){
        Proxy.Metadata metadata = packet.getHeader();
        Transfer.RollSiteHeader rollSiteHeader = null;
        try {
            rollSiteHeader = Transfer.RollSiteHeader.parseFrom(metadata.getExt());
        } catch (InvalidProtocolBufferException e) {
            throw new ParameterException("invalid rollSiteHeader");
        }
        String dstPartyId = rollSiteHeader.getDstPartyId();
        if (StringUtils.isEmpty(dstPartyId)) {
            dstPartyId = metadata.getDst().getPartyId();
        }

        String desRole = metadata.getDst().getRole();
        String srcRole = metadata.getSrc().getRole();
        String srcPartyId = metadata.getSrc().getPartyId();
        context.setSrcPartyId(srcPartyId);
        context.setDesPartyId(dstPartyId);
        context.setSrcComponent(srcRole);
        context.setDesComponent(desRole);
    }

}
