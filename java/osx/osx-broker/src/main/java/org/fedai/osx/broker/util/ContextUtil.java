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
package org.fedai.osx.broker.util;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;

import org.fedai.osx.core.config.MetaInfo;
import org.fedai.osx.core.constant.PtpHttpHeader;
import org.fedai.osx.core.context.OsxContext;
import org.fedai.osx.core.context.Protocol;

import javax.servlet.http.HttpServletRequest;
import java.util.Enumeration;

import static org.fedai.osx.core.frame.ContextPrepareInterceptor.*;
import static org.fedai.osx.core.frame.ContextPrepareInterceptor.CONTEXTKEY_TOPIC_KEY;
@Slf4j
public class ContextUtil {


    public static void assableContextFromInbound(OsxContext context) {

        io.grpc.Context  grpcContext  = io.grpc.Context.current();



//            .withValue(CONTEXTKEY_TRACE_ID, Optional.ofNullable(metadata.get(METAKEY_TRACE_ID)).orElse(""))
//                .withValue(CONTEXTKEY_FROM_INST_ID, metadata.get(METAKEY_FROM_INST_ID))
//                .withValue(CONTEXTKEY_FROM_NODE_ID, metadata.get(METAKEY_FROM_NODE_ID))
//                .withValue(CONTEXTKEY_VERSION, metadata.get(METAKEY_VERSION))
////                .withValue(tim metadata.get(GrpcContextKey.TIMESTAMP))
//                .withValue(CONTEXTKEY_SOURCEIP, remoteIp)
//                .withValue(CONTEXTKEY_TECH_PROVIDER, metadata.get(METAKEY_TECH_PROVIDER_CODE))
//                .withValue(CONTEXTKEY_TOKEN, metadata.get(METAKEY_TOKEN))
//                .withValue(CONTEXTKEY_TARGET_NODE_ID, metadata.get(METAKEY_TARGET_NODE_ID))
//                .withValue(CONTEXTKEY_TARGET_INST_ID, metadata.get(METAKEY_TARGET_INST_ID))
//                .withValue(CONTEXTKEY_SESSION_ID, metadata.get(METAKEY_SESSION_ID));
//        context.setTraceId(CONTEXTKEY_TRACE_ID.get(grpcContext));
//        context.setSourceInstId(CONTEXTKEY_FROM_INST_ID.get(grpcContext));
//        context.setSourceNodeId(CONTEXTKEY_FROM_NODE_ID.get(grpcContext));
//        context.setDesInstId(CONTEXTKEY_TARGET_INST_ID.get(grpcContext));
//        context.setDesNodeId(CONTEXTKEY_SESSION_ID.get(grpcContext));
//        context.setVersion(CONTEXTKEY_VERSION.get(grpcContext));
//        context.setTechProviderCode(CONTEXTKEY_TECH_PROVIDER.get(grpcContext));
//        context.setToken(CONTEXTKEY_TOKEN.get(grpcContext));
//        context.setSessionId(CONTEXTKEY_SESSION_ID.get(grpcContext));


//        String version = metaDataMap.get(Osx.Header.Version.name());
//        String jobId = metaDataMap.get(Osx.Metadata.JobId.name());
        String techProviderCode = CONTEXTKEY_TECH_PROVIDER.get(grpcContext);
        String traceId = CONTEXTKEY_TRACE_ID.get(grpcContext);
        String token = CONTEXTKEY_TOKEN.get(grpcContext);
        String sourceNodeId = CONTEXTKEY_FROM_NODE_ID.get(grpcContext);
        String targetNodeId = CONTEXTKEY_TARGET_NODE_ID.get(grpcContext);
        String sourceInstId = CONTEXTKEY_FROM_INST_ID.get(grpcContext);
        String targetInstId = CONTEXTKEY_TARGET_INST_ID.get(grpcContext);
        String sessionId = CONTEXTKEY_SESSION_ID.get(grpcContext);
        String queueType = CONTEXTKEY_QUEUE_TYPE.get(grpcContext);
        String msgFlag = CONTEXTKEY_MSG_FLAG.get(grpcContext);
//        String targetMethod = metaDataMap.get(Osx.Metadata.TargetMethod.name());
//        String targetComponentName = metaDataMap.get(Osx.Metadata.TargetComponentName.name());
//        String sourceComponentName = metaDataMap.get(Osx.Metadata.SourceComponentName.name());
        String uri = CONTEXTKEY_URI.get(grpcContext);
        String sourcePartyId = StringUtils.isEmpty(sourceInstId) ? sourceNodeId : sourceInstId + "." + sourceNodeId;
        String targetPartyId = StringUtils.isEmpty(targetInstId) ? targetNodeId : targetInstId + "." + targetNodeId;
        String topic = CONTEXTKEY_TOPIC_KEY.get(grpcContext);
//        String offsetString = metaDataMap.get(Osx.Metadata.MessageOffSet.name());
//        String messageCode = metaDataMap.get(Osx.Metadata.MessageCode.name());
//        Long offset = StringUtils.isNotEmpty(offsetString) ? Long.parseLong(offsetString) : null;
        context.setTraceId(traceId);
        context.setToken(token);
        context.setDesNodeId(targetNodeId);
        context.setDesInstId(targetInstId);
        context.setSrcInstId(sourceInstId);

        context.setSrcNodeId(sourceNodeId);
        context.setDesNodeId(targetPartyId);

        context.setTopic(topic);
        context.setQueueType(queueType);
        context.setMessageFlag(msgFlag);
//        context.setJobId(jobId);
        context.setTopic(topic);
        context.setUri(uri);
        context.setSessionId(sessionId);
//        context.setRequestMsgIndex(offset);
//        context.setMessageCode(messageCode);
        context.setTechProviderCode(techProviderCode);
        if (MetaInfo.PROPERTY_SELF_PARTY.contains(context.getDesNodeId())) {
            context.setSelfPartyId(context.getDesNodeId());
        } else {
            context.setSelfPartyId(MetaInfo.PROPERTY_SELF_PARTY.toArray()[0].toString());
        }
    }

    public static OsxContext buildFateContext(Protocol protocol) {
        OsxContext context = new OsxContext();
        context.setProtocol(protocol);
      //  context.setSourceIp(ContextPrepareInterceptor.sourceIp.get() != null ? ContextPrepareInterceptor.sourceIp.get().toString() : "");

        return context;
    }

    public static  OsxContext  buildContextFromHttpRequest(HttpServletRequest  request){
        OsxContext  osxContext = new OsxContext();
        String version = request.getHeader(PtpHttpHeader.Version);
        String techProviderCode = request.getHeader(PtpHttpHeader.TechProviderCode);
        String traceID = request.getHeader(PtpHttpHeader.TraceID);
        String token = request.getHeader(PtpHttpHeader.Token);
        String sourceNodeID = request.getHeader(PtpHttpHeader.FromNodeID);
        String targetNodeID = request.getHeader(PtpHttpHeader.TargetNodeID);
        String sourceInstID = request.getHeader(PtpHttpHeader.FromInstID);
        String targetInstID = request.getHeader(PtpHttpHeader.TargetInstID);
        String sessionID = request.getHeader(PtpHttpHeader.SessionID);
        String uri =   request.getHeader(PtpHttpHeader.Uri);
        String topic = request.getHeader(PtpHttpHeader.MessageTopic);
        String msgFlag = request.getHeader(PtpHttpHeader.MessageFlag);
        String queueType = request.getHeader(PtpHttpHeader.QueueType);

        Enumeration<String>   headers =  request.getHeaderNames();
        while(headers.hasMoreElements()){
            String name = headers.nextElement();
            log.info("==http head======"+name+"======="+request.getHeader(name));
        }


        osxContext.setUri(uri);
        osxContext.setVersion(version);
        osxContext.setTraceId(traceID);
        osxContext.setTopic(topic);
        osxContext.setTechProviderCode(techProviderCode);
        osxContext.setToken(token);

        osxContext.setSrcNodeId(sourceNodeID);
        osxContext.setDesNodeId(targetNodeID);

        osxContext.setSessionId(sessionID);
        osxContext.setDesInstId(targetInstID);
        osxContext.setQueueType(queueType);
        osxContext.setMessageFlag(msgFlag);
        return  osxContext;
    }


}
