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
package com.osx.tech.provider;


import com.google.common.collect.Sets;
import com.google.protobuf.ByteString;
import com.osx.api.context.Context;
import com.osx.broker.interceptor.PcpHandleInterceptor;
import com.osx.broker.interceptor.TokenValidatorInterceptor;
import com.osx.broker.router.RouterRegister;
import com.osx.broker.util.ContextUtil;
import com.osx.broker.interceptor.RouterInterceptor;
import com.osx.broker.ptp.*;
import com.osx.broker.util.TransferUtil;
import com.osx.core.config.MetaInfo;
import com.osx.core.constant.Dict;
import com.osx.api.constants.Protocol;
import com.osx.core.constant.PtpHttpHeader;

import com.osx.core.exceptions.ErrorMessageUtil;
import com.osx.core.exceptions.ExceptionInfo;
import com.osx.core.exceptions.ParameterException;
import com.osx.core.provider.TechProvider;
import com.osx.core.ptp.TargetMethod;
import com.osx.core.service.InboundPackage;
import com.osx.core.service.OutboundPackage;
import com.osx.core.service.ServiceAdaptor;
import com.osx.core.utils.FlowLogUtil;
import io.grpc.stub.StreamObserver;
import org.apache.commons.io.IOUtils;
import org.apache.commons.lang3.StringUtils;
import org.ppc.ptp.Osx;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.*;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;

/**
 * FATE 相关实现
 */

public class FateTechProvider implements TechProvider {

    Logger logger = LoggerFactory.getLogger(FateTechProvider.class);
    ConcurrentMap<String, ServiceAdaptor> serviceAdaptorConcurrentMap = new ConcurrentHashMap<>();
    PcpHandleInterceptor requestHandleInterceptor;
    TokenValidatorInterceptor tokenValidatorInterceptor;
    RouterInterceptor routerInterceptor;
    private Set<String> httpAllowedMethod = Sets.newHashSet(TargetMethod.PRODUCE_MSG.name(), TargetMethod.UNARY_CALL.name());

    public FateTechProvider() {
        requestHandleInterceptor = new PcpHandleInterceptor();
        tokenValidatorInterceptor = new TokenValidatorInterceptor();
        routerInterceptor = new RouterInterceptor();
        registerServiceAdaptor();
    }

    private void checkHttpAllowedMethod(String targetMethod) {
        if (!httpAllowedMethod.contains(targetMethod)) {
            throw new ParameterException("target method :" + targetMethod + "is not allowed");
        }
    }

    @Override
    public void processHttpInvoke(HttpServletRequest request, HttpServletResponse response) {
        Context context = ContextUtil.buildFateContext(Protocol.http);
        context.putData(Dict.HTTP_SERVLET_RESPONSE, response);
        Osx.Inbound.Builder inboundBuilder;
        ServiceAdaptor serviceAdaptor = null;
        try {
            inboundBuilder = TransferUtil.buildPbFromHttpRequest(context, request);
            String targetMethod = inboundBuilder.getMetadataMap().get(Osx.Metadata.TargetMethod.name());
            if (StringUtils.isEmpty(targetMethod)) {
                throw new ParameterException("target method is null");
            }
            checkHttpAllowedMethod(targetMethod);
            serviceAdaptor = this.getServiceAdaptor(targetMethod);

            byte[] buffer = new byte[MetaInfo.PROPERTY_HTTP_REQUEST_BODY_MAX_SIZE];
            int length = IOUtils.read(request.getInputStream(), buffer);
            byte[] data = new byte[length];
            System.arraycopy(buffer, 0, data, 0, length);
            inboundBuilder.setPayload(ByteString.copyFrom(data));
        } catch (Exception e) {
            ExceptionInfo exceptionInfo = ErrorMessageUtil.handleExceptionExceptionInfo(context, e);
            this.writeHttpRespose(response, exceptionInfo.getCode(), exceptionInfo.getMessage(), null);
            context.setReturnCode(exceptionInfo.getCode());
            context.setReturnMsg(exceptionInfo.getMessage());
            FlowLogUtil.printFlowLog(context);
            return;
        }
        InboundPackage inboundPackage = new InboundPackage();
        inboundPackage.setBody(inboundBuilder.build());
        OutboundPackage<Osx.Outbound> outboundPackage = serviceAdaptor.service(context, inboundPackage);
        Osx.Outbound outbound = outboundPackage.getData();
        response.setContentType(Dict.CONTENT_TYPE_JSON_UTF8);
        TransferUtil.writeHttpRespose(response, outbound.getCode(), outbound.getMessage(), outbound.getPayload().toByteArray());
    }

    private void writeHttpRespose(HttpServletResponse response, String code,
                                  String msg,
                                  byte[] content) {
        try {
            response.setHeader(PtpHttpHeader.ReturnCode, code);
            response.setHeader(PtpHttpHeader.MessageCode, msg);
            OutputStream outputStream = response.getOutputStream();
            if (content != null) {
                outputStream.write(content);
            }
            outputStream.flush();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }


    @Override
    public void processGrpcInvoke(Osx.Inbound request, StreamObserver<Osx.Outbound> responseObserver) {
        Context context = ContextUtil.buildFateContext(Protocol.grpc);
        context.putData(Dict.RESPONSE_STREAM_OBSERVER, responseObserver);
        Osx.Outbound result = null;
        try {
            Map<String, String> metaDataMap = request.getMetadataMap();
            String targetMethod = metaDataMap.get(Osx.Metadata.TargetMethod.name());
            ServiceAdaptor serviceAdaptor = this.getServiceAdaptor(targetMethod);
            if (serviceAdaptor == null) {
                throw new ParameterException("invalid target method " + targetMethod);
            }
            InboundPackage inboundPackage = new InboundPackage();
            inboundPackage.setBody(request);
            OutboundPackage<Osx.Outbound> outboundPackage = serviceAdaptor.service(context, inboundPackage);
            if (outboundPackage.getData() != null) {
                result = outboundPackage.getData();
            }
        } catch (Exception e) {
            ExceptionInfo exceptionInfo = ErrorMessageUtil.handleExceptionExceptionInfo(context, e);
            //this.writeHttpRespose(response, exceptionInfo.getCode(),exceptionInfo.getMessage(),null);
            context.setReturnCode(exceptionInfo.getCode());
            context.setReturnMsg(exceptionInfo.getMessage());
            FlowLogUtil.printFlowLog(context);
            result = Osx.Outbound.newBuilder().setCode(exceptionInfo.getCode()).setMessage(exceptionInfo.getMessage()).build();
        }
        if (result != null) {
            responseObserver.onNext(result);
            responseObserver.onCompleted();
        }

    }


    @Override
    public StreamObserver<Osx.Inbound> processGrpcTransport(Osx.Inbound fristPackage, StreamObserver<Osx.Outbound> responseObserver) {
        Map<String, String> metaDataMap = fristPackage.getMetadataMap();
        String targetMethod = metaDataMap.get(Osx.Metadata.TargetMethod.name());
        ServiceAdaptor serviceAdaptor = this.getServiceAdaptor(targetMethod);
        if (serviceAdaptor == null) {
            throw new ParameterException("invalid target method " + targetMethod);
        }
        Context context = ContextUtil.buildFateContext(Protocol.grpc);
        InboundPackage inboundPackage = new InboundPackage();
        inboundPackage.setBody(responseObserver);
        OutboundPackage<StreamObserver<Osx.Inbound>> outboundPackage = serviceAdaptor.service(context, inboundPackage);
        if (outboundPackage != null && outboundPackage.getData() != null) {
            return (StreamObserver<Osx.Inbound>) outboundPackage.getData();
        } else {
            return null;
        }
    }

    @Override
    public void processGrpcPeek(Osx.PeekInbound inbound, StreamObserver<Osx.TransportOutbound> responseObserver) {

    }

    @Override
    public void processGrpcPush(Osx.PushInbound inbound, StreamObserver<Osx.TransportOutbound> responseObserver) {

    }

    @Override
    public void processGrpcPop(Osx.PopInbound inbound, StreamObserver<Osx.TransportOutbound> responseObserver) {

    }

    @Override
    public void processGrpcRelease(Osx.ReleaseInbound inbound, StreamObserver<Osx.TransportOutbound> responseObserver) {

    }


    public ServiceAdaptor getServiceAdaptor(String name) {
        return this.serviceAdaptorConcurrentMap.get(name);
    }

    private void registerServiceAdaptor() {
        this.serviceAdaptorConcurrentMap.put(TargetMethod.UNARY_CALL.name(), new PtpUnaryCallService()
                .addPreProcessor(requestHandleInterceptor)
                .addPreProcessor(tokenValidatorInterceptor)
                .addPreProcessor(routerInterceptor));
        this.serviceAdaptorConcurrentMap.put(TargetMethod.PRODUCE_MSG.name(), new PtpProduceService()
                .addPreProcessor(requestHandleInterceptor)
                .addPreProcessor(routerInterceptor));
        this.serviceAdaptorConcurrentMap.put(TargetMethod.ACK_MSG.name(), new PtpAckService()
                .addPreProcessor(requestHandleInterceptor));
        this.serviceAdaptorConcurrentMap.put(TargetMethod.CONSUME_MSG.name(), new PtpConsumeService()
                .addPreProcessor(requestHandleInterceptor));
        this.serviceAdaptorConcurrentMap.put(TargetMethod.QUERY_TOPIC.name(), new PtpQueryTransferQueueService()
                .addPreProcessor(requestHandleInterceptor));
        this.serviceAdaptorConcurrentMap.put(TargetMethod.CANCEL_TOPIC.name(), new PtpCancelTransferService()
                .addPreProcessor(requestHandleInterceptor));
        this.serviceAdaptorConcurrentMap.put(TargetMethod.PUSH.name(), new PtpPushService());
        this.serviceAdaptorConcurrentMap.put(TargetMethod.APPLY_TOKEN.name(), new PtpClusterTokenApplyService());
        this.serviceAdaptorConcurrentMap.put(TargetMethod.APPLY_TOPIC.name(), new PtpClusterTopicApplyService());
        // this.serviceAdaptorConcurrentMap.put(TargetMethod.TEST_STREAM.name(), new  PtpStreamTestService());
    }
}
