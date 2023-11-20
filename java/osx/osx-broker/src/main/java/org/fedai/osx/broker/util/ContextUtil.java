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
import org.fedai.osx.core.utils.UrlUtil;

import javax.servlet.http.HttpServletRequest;
import java.util.Enumeration;

import static org.fedai.osx.core.frame.ContextPrepareInterceptor.*;

@Slf4j
public class ContextUtil {


    public static void assableContextFromInbound(OsxContext context) {
        io.grpc.Context grpcContext = io.grpc.Context.current();
        String techProviderCode = CONTEXTKEY_TECH_PROVIDER.get(grpcContext);
        String traceId = CONTEXTKEY_TRACE_ID.get(grpcContext);
        String token = CONTEXTKEY_TOKEN.get(grpcContext);
        String sourceNodeId = CONTEXTKEY_FROM_NODE_ID.get(grpcContext);
        String targetNodeId = CONTEXTKEY_TARGET_NODE_ID.get(grpcContext);
        String sourceInstId = CONTEXTKEY_FROM_INST_ID.get(grpcContext);
        if (StringUtils.isEmpty(sourceNodeId)) {
            sourceNodeId = sourceInstId;
        }
        String targetInstId = CONTEXTKEY_TARGET_INST_ID.get(grpcContext);
        if (StringUtils.isEmpty(targetNodeId)) {
            targetNodeId = targetInstId;
        }
        String sessionId = CONTEXTKEY_SESSION_ID.get(grpcContext);
        String queueType = CONTEXTKEY_QUEUE_TYPE.get(grpcContext);
        String msgFlag = CONTEXTKEY_MSG_FLAG.get(grpcContext);
        String uri = CONTEXTKEY_URI.get(grpcContext);
        String sourcePartyId = sourceNodeId;
        String targetPartyId = targetNodeId;
        String topic = CONTEXTKEY_TOPIC_KEY.get(grpcContext);
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

    public static OsxContext buildContextFromHttpRequest(HttpServletRequest request) {
        OsxContext osxContext = new OsxContext();
        String version = request.getHeader(PtpHttpHeader.Version);
//        System.err.println("version :" +version);
        String techProviderCode = request.getHeader(PtpHttpHeader.TechProviderCode);
//        System.err.println("techProviderCode :" +techProviderCode);
        String traceID = request.getHeader(PtpHttpHeader.TraceID);
//        System.err.println("TraceID :" +traceID);
        String token = request.getHeader(PtpHttpHeader.Token);
//        System.err.println("Token :" +token);
        String sourceNodeID = request.getHeader(PtpHttpHeader.FromNodeID);
//        System.err.println("FromNodeID :" +sourceNodeID);
        String targetNodeID = request.getHeader(PtpHttpHeader.TargetNodeID);
//        System.err.println("TargetNodeID :" +targetNodeID);
        String sourceInstID = request.getHeader(PtpHttpHeader.FromInstID);
        String targetInstID = request.getHeader(PtpHttpHeader.TargetInstID);
        String sessionID = request.getHeader(PtpHttpHeader.SessionID);
        System.err.println("sessionId :" + sessionID);
        String uri = request.getHeader(PtpHttpHeader.Uri);
        String topic = request.getHeader(PtpHttpHeader.MessageTopic);
        String msgFlag = request.getHeader(PtpHttpHeader.MessageFlag);
        String queueType = request.getHeader(PtpHttpHeader.QueueType);
        Enumeration<String> headers = request.getHeaderNames();
//        while(headers.hasMoreElements()){
//            String name = headers.nextElement();
//            log.info("==http head======"+name+"======="+request.getHeader(name));
//        }
        osxContext.setUri(UrlUtil.parseUri(uri));
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
        System.err.println("xxxxxxxxxxx+" + osxContext.toString());
        return osxContext;
    }


}
