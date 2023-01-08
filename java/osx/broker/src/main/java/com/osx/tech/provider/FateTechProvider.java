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


import com.google.common.base.Preconditions;
import com.google.common.collect.Sets;
import com.google.protobuf.ByteString;
import com.osx.broker.ServiceContainer;
import com.osx.broker.util.ContextUtil;
import com.osx.broker.interceptor.RequestHandleInterceptor;
import com.osx.broker.interceptor.RouterInterceptor;
import com.osx.broker.ptp.*;
import com.osx.broker.util.TransferUtil;
import com.osx.core.config.MetaInfo;
import com.osx.core.constant.Dict;
import com.osx.core.constant.PtpHttpHeader;
import com.osx.core.context.Context;
import com.osx.core.exceptions.ErrorMessageUtil;
import com.osx.core.exceptions.ExceptionInfo;
import com.osx.core.exceptions.ParameterException;
import com.osx.core.frame.Lifecycle;
import com.osx.core.provider.TechProvider;
import com.osx.core.ptp.TargetMethod;
import com.osx.core.service.InboundPackage;
import com.osx.core.service.OutboundPackage;
import com.osx.core.service.ServiceAdaptor;
import io.grpc.stub.StreamObserver;
import org.apache.commons.io.IOUtils;
import org.ppc.ptp.Osx;


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

public class FateTechProvider implements TechProvider, Lifecycle {

    ConcurrentMap<String, ServiceAdaptor> serviceAdaptorConcurrentMap = new ConcurrentHashMap<>();

    RequestHandleInterceptor requestHandleInterceptor;
    RouterInterceptor  routerInterceptor;

    private Set<String> httpAllowedMethod= Sets.newHashSet(TargetMethod.PRODUCE_MSG.name(),TargetMethod.UNARY_CALL.name());

    private  void  checkHttpAllowedMethod(String  targetMethod){

        if(!httpAllowedMethod.contains(targetMethod)){
            throw  new ParameterException("target method :"+targetMethod+"is not allowed");
        }

    }

    @Override
    public void processHttpInvoke(HttpServletRequest request, HttpServletResponse response) {
        Context context = ContextUtil.buildContext();
        Osx.Inbound.Builder inboundBuilder ;
        ServiceAdaptor serviceAdaptor=null;
        try {
            String Version = request.getHeader(PtpHttpHeader.Version);
            String TechProviderCode = request.getHeader(PtpHttpHeader.TechProviderCode);
            String TraceID = request.getHeader(PtpHttpHeader.TraceID);
            String Token = request.getHeader(PtpHttpHeader.Token);
            String SourceNodeID = request.getHeader(PtpHttpHeader.SourceNodeID);
            String TargetNodeID = request.getHeader(PtpHttpHeader.TargetNodeID);
            String SourceInstID = request.getHeader(PtpHttpHeader.SourceInstID);
            String TargetInstID = request.getHeader(PtpHttpHeader.TargetInstID);
            String SessionID = request.getHeader(PtpHttpHeader.SessionID);
            String MessageTopic = request.getHeader(PtpHttpHeader.MessageTopic);
            String MessageCode = request.getHeader(PtpHttpHeader.MessageCode);
            String SourceComponentName = request.getHeader(PtpHttpHeader.SourceComponentName);
            String TargetComponentName = request.getHeader(PtpHttpHeader.TargetComponentName);
            String TargetMethod = request.getHeader(PtpHttpHeader.TargetMethod);
            String MessageOffSet = request.getHeader(PtpHttpHeader.MessageOffSet);
            String InstanceId = request.getHeader(PtpHttpHeader.InstanceId);
            String Timestamp = request.getHeader(PtpHttpHeader.Timestamp);
            context.setSrcPartyId(SourceNodeID);
            context.setDesPartyId(TargetNodeID);
            context.setSessionId(SessionID);
            context.setTopic(MessageTopic);
            context.setActionType(TargetMethod);
            inboundBuilder = TransferUtil.buildPbFromHttpRequest(request);
            String targetMethod = inboundBuilder.getMetadataMap().get(Osx.Metadata.TargetMethod.name());
            checkHttpAllowedMethod(TargetMethod);
            serviceAdaptor = this.getServiceAdaptor(TargetMethod);
            byte[] buffer = new byte[MetaInfo.PROPERTY_HTTP_REQUEST_BODY_MAX_SIZE];
            int length = IOUtils.read(request.getInputStream(), buffer);
            byte[] data = new byte[length];
            System.arraycopy(buffer, 0, data, 0, length);
            inboundBuilder.setPayload(ByteString.copyFrom(data));
        }catch(Exception e){
            ExceptionInfo exceptionInfo =  ErrorMessageUtil.handleExceptionExceptionInfo(context,e);
            this.writeHttpRespose(response, exceptionInfo.getCode(),exceptionInfo.getMessage(),null);
            context.setReturnCode(exceptionInfo.getCode());
            context.setReturnMsg(exceptionInfo.getMessage());
            context.printFlowLog();
            return ;
        }
            InboundPackage inboundPackage = new InboundPackage();
            inboundPackage.setBody(inboundBuilder.build());
            OutboundPackage<Osx.Outbound> outboundPackage = serviceAdaptor.service(context, inboundPackage);
            Osx.Outbound outbound = outboundPackage.getData();
            response.setContentType(Dict.CONTENT_TYPE_JSON_UTF8);
            this.writeHttpRespose(response,outbound.getCode(),outbound.getMessage(),outbound.getPayload().toByteArray() );
    }

    private  void  writeHttpRespose(HttpServletResponse  response,String code,
                                    String msg,
                                    byte[] content){
        try {
            response.setHeader(PtpHttpHeader.ReturnCode,code);
            response.setHeader(PtpHttpHeader.MessageCode,msg);
            OutputStream  outputStream = response.getOutputStream();
            if(content!=null) {
                outputStream.write(content);
            }
            outputStream.flush();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }


    @Override
    public void processGrpcInvoke(Osx.Inbound request, StreamObserver<Osx.Outbound> responseObserver) {
        Context context = ContextUtil.buildContext();
        context.putData(Dict.RESPONSE_STREAM_OBSERVER,responseObserver);
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
        }catch (Exception e){
            ExceptionInfo exceptionInfo =  ErrorMessageUtil.handleExceptionExceptionInfo(context,e);
            //this.writeHttpRespose(response, exceptionInfo.getCode(),exceptionInfo.getMessage(),null);
            context.setReturnCode(exceptionInfo.getCode());
            context.setReturnMsg(exceptionInfo.getMessage());
            context.printFlowLog();
            result = Osx.Outbound.newBuilder().setCode(exceptionInfo.getCode()).setMessage(exceptionInfo.getMessage()).build();
        }
        if(result!=null) {
            responseObserver.onNext(result);
            responseObserver.onCompleted();
        }

    }

    @Override
    public String getProviderId() {
        return Dict.FATE_TECH_PROVIDER;
    }


    @Override
    public StreamObserver<Osx.Inbound> processGrpcTransport(Osx.Inbound fristPackage, StreamObserver<Osx.Outbound> responseObserver) {
        Map<String, String> metaDataMap = fristPackage.getMetadataMap();
        String targetMethod = metaDataMap.get(Osx.Metadata.TargetMethod.name());
        ServiceAdaptor serviceAdaptor = this.getServiceAdaptor(targetMethod);
        if(serviceAdaptor==null){
            throw new ParameterException("invalid target method "+targetMethod);
        }
        Context context = ContextUtil.buildContext();
        InboundPackage inboundPackage = new InboundPackage();
        inboundPackage.setBody(responseObserver);
        OutboundPackage<StreamObserver<Osx.Inbound>> outboundPackage = serviceAdaptor.service( context, inboundPackage);
        if(outboundPackage!=null&&outboundPackage.getData()!=null){
            return (StreamObserver<Osx.Inbound>)outboundPackage.getData();
        }else{
            return null;
        }


    }

    @Override
    public void init() {
        Preconditions.checkArgument(ServiceContainer.fateRouterService != null);
        requestHandleInterceptor = new RequestHandleInterceptor();
        routerInterceptor = new RouterInterceptor(ServiceContainer.fateRouterService);
        registerServiceAdaptor();
    }

    @Override
    public void start() {

    }

    @Override
    public void destroy() {

    }
    public ServiceAdaptor getServiceAdaptor(String name) {
        return this.serviceAdaptorConcurrentMap.get(name);
    }
    private void registerServiceAdaptor() {
        this.serviceAdaptorConcurrentMap.put(TargetMethod.UNARY_CALL.name(), new PtpUnaryCallService().addPreProcessor(requestHandleInterceptor)
                .addPreProcessor(routerInterceptor));
        this.serviceAdaptorConcurrentMap.put(TargetMethod.PRODUCE_MSG.name(), new PtpProduceService().addPreProcessor(requestHandleInterceptor)
                .addPreProcessor(routerInterceptor));
        this.serviceAdaptorConcurrentMap.put(TargetMethod.ACK_MSG.name(), new PtpAckService().addPreProcessor(requestHandleInterceptor));
        this.serviceAdaptorConcurrentMap.put(TargetMethod.CONSUME_MSG.name(), new PtpConsumeService().addPreProcessor(requestHandleInterceptor));
        this.serviceAdaptorConcurrentMap.put(TargetMethod.QUERY_TOPIC.name(), new PtpQueryTransferQueueService().addPreProcessor(requestHandleInterceptor));
        this.serviceAdaptorConcurrentMap.put(TargetMethod.CANCEL_TOPIC.name(), new PtpCancelTransferService().addPreProcessor(requestHandleInterceptor));
        this.serviceAdaptorConcurrentMap.put(TargetMethod.PUSH.name(), new PtpPushService());
        this.serviceAdaptorConcurrentMap.put(TargetMethod.APPLY_TOKEN.name(), new PtpClusterTokenApplyService());
        this.serviceAdaptorConcurrentMap.put(TargetMethod.APPLY_TOPIC.name(),new PtpClusterTopicApplyService());
    }
}
