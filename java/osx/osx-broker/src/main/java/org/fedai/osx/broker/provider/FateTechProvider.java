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
package org.fedai.osx.broker.provider;


import com.google.common.collect.Lists;
import com.google.inject.Inject;
import com.google.inject.Singleton;
import com.google.protobuf.ByteString;
import io.grpc.stub.StreamObserver;

import org.fedai.osx.broker.constants.ServiceType;
import org.fedai.osx.broker.pojo.ConsumeRequest;
import org.fedai.osx.broker.pojo.ConsumerResponse;
import org.fedai.osx.broker.pojo.ProduceRequest;
import org.fedai.osx.broker.pojo.ProduceResponse;
import org.fedai.osx.broker.router.DefaultFateRouterServiceImpl;
import org.fedai.osx.broker.service.ServiceRegisterInfo;
import org.fedai.osx.broker.service.ServiceRegisterManager;
import org.fedai.osx.broker.util.TransferUtil;
import org.fedai.osx.core.config.MetaInfo;
import org.fedai.osx.core.constant.Dict;
import org.fedai.osx.core.constant.PtpHttpHeader;
import org.fedai.osx.core.constant.UriConstants;
import org.fedai.osx.core.context.OsxContext;
import org.fedai.osx.core.context.Protocol;
import org.fedai.osx.core.exceptions.*;
import org.fedai.osx.core.provider.TechProvider;
import org.fedai.osx.core.router.RouterInfo;
import org.fedai.osx.core.service.*;
import org.fedai.osx.core.utils.FlowLogUtil;
import org.ppc.ptp.Osx;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.List;

import static org.fedai.osx.core.constant.ActionType.MSG_REDIRECT;

/**
 * FATE 相关实现
 */
@Singleton
public class FateTechProvider implements TechProvider {

    Logger logger = LoggerFactory.getLogger(FateTechProvider.class);
    @Inject
    ServiceRegisterManager serviceRegisterManager;
    @Inject
    DefaultFateRouterServiceImpl routerService;

    List<String> selfPartyIds;

    public FateTechProvider() {
        selfPartyIds = Lists.newArrayList(MetaInfo.PROPERTY_SELF_PARTY);

    }



    @Override
    public void processHttpInvoke(OsxContext  osxContext , HttpServletRequest request, HttpServletResponse response) {

        try {
            osxContext.setProtocol(Protocol.http);
            osxContext.putData(Dict.HTTP_SERVLET_RESPONSE, response);
            OsxContext.pushThreadLocalContext(osxContext);
            Osx.Inbound.Builder  inboundBuilder = Osx.Inbound.newBuilder();
            byte[] payload = this.read(request.getInputStream());
            System.err.println("receive"+new String(payload));
            inboundBuilder.setPayload(ByteString.copyFrom(payload));
            Osx.Outbound   outbound = this.handleInner(osxContext,inboundBuilder.build());

            logger.info("processHttpInvoke outbound {}",outbound);
            response.setContentType(Dict.CONTENT_TYPE_JSON_UTF8);
            TransferUtil.writeHttpRespose(response, outbound.getCode(), outbound.getMessage(), outbound.toByteArray());
        } catch (IOException e) {
            e.printStackTrace();
        }
    }




    @Override
    public void processGrpcInvoke(OsxContext  context,Osx.Inbound request, StreamObserver<Osx.Outbound> responseObserver) {
        context.setProtocol(Protocol.grpc);
        context.putData(Dict.RESPONSE_STREAM_OBSERVER, responseObserver);
        OsxContext.pushThreadLocalContext(context);
        Osx.Outbound result = handleInner(context,request);;
        if (result != null) {
            responseObserver.onNext(result);
            responseObserver.onCompleted();
        }

    }


//    @Override
//    public StreamObserver<Osx.Inbound> processGrpcTransport(OsxContext context,Osx.Inbound fristPackage, StreamObserver<Osx.Outbound> responseObserver) {
//        Map<String, String> metaDataMap = fristPackage.getMetadataMap();
//        String targetMethod = metaDataMap.get(Osx.Metadata.TargetMethod.name());
//        String uri = context.getUri();
//        ServiceAdaptor serviceAdaptor = this.serviceRegister
//        if (serviceAdaptor == null) {
//            throw new ParameterException("invalid target method " + targetMethod);
//        }
////        Context context = ContextUtil.buildFateContext(Protocol.grpc);
//        context.setProtocol(Protocol.grpc);
//        InboundPackage inboundPackage = new InboundPackage();
//        inboundPackage.setBody(responseObserver);
//        OutboundPackage<StreamObserver<Osx.Inbound>> outboundPackage = serviceAdaptor.service(context, inboundPackage);
//        if (outboundPackage != null && outboundPackage.getData() != null) {
//            return (StreamObserver<Osx.Inbound>) outboundPackage.getData();
//        } else {
//            return null;
//        }
//    }

    private  Osx.Outbound  handleInner(OsxContext  context, Osx.Inbound  request){
        Osx.Outbound  result = null;
        try {
            String uri = context.getUri();
            String instId = context.getDesInstId();
            String nodeId = context.getDesNodeId();
            if(MetaInfo.PROPERTY_SELF_PARTY.contains(nodeId)){
                ServiceRegisterInfo serviceRegisterInfo = this.serviceRegisterManager.getServiceWithLoadBalance(context,nodeId,uri,true);
                if(serviceRegisterInfo!=null){
                    ServiceType serviceType =  serviceRegisterInfo.getServiceType();
                    switch (serviceType){
                        case  inner:
                            Object serviceAdaptorObject = serviceRegisterInfo.getServiceAdaptor();
                            ServiceAdaptorNew   serviceAdaptor = (ServiceAdaptorNew) serviceAdaptorObject;
                            Object  requestObj = serviceAdaptor.decode(request);
                            Object  responseObj = serviceAdaptor.service(context,requestObj );
                            result =  serviceAdaptor.toOutbound(responseObj);
                            break;
                        case  rpc:
                            RouterInfo routerInfo = serviceRegisterInfo.getRouterInfo();
                            result = TransferUtil.redirect(context,request,routerInfo,true);
                            break;
                    }
                }else {
                    logger.error("invalid  uri {}",uri);
                    throw  new InvalidUriException();
                }
            }

            else{
                //外部则转发
                // RouterService routerService = routerRegister.getRouterService(MetaInfo.PROPERTY_FATE_TECH_PROVIDER);

                RouterInfo routerInfo = routerService.routePtp(context.getSrcInstId(),context.getSrcNodeId(),context.getDesInstId(),context.getDesNodeId());
                if(routerInfo!=null) {
                    result  =  TransferUtil.redirect(context, request, routerInfo, true);
                }else{
                    logger.error("can not found router info {} {}",context.getDesInstId(),context.getDesNodeId());
                    throw  new NoRouterInfoException("can not found router info");
                }
            }
        } catch (Exception e) {
            logger.error("===========error =========",e);
            ExceptionInfo exceptionInfo = ErrorMessageUtil.handleExceptionExceptionInfo(context, e);
            //this.writeHttpRespose(response, exceptionInfo.getCode(),exceptionInfo.getMessage(),null);
            context.setReturnCode(exceptionInfo.getCode());
            context.setReturnMsg(exceptionInfo.getMessage());

            result = Osx.Outbound.newBuilder().setCode(exceptionInfo.getCode()).setMessage(exceptionInfo.getMessage()).build();
        }
        finally {
            FlowLogUtil.printFlowLog(context);
            OsxContext.popThreadLocalContext();
            OsxContext.release();
        }

        return  result;

    }



    @Override
    public void processGrpcPeek(OsxContext  context ,Osx.PeekInbound inbound, StreamObserver<Osx.TransportOutbound> responseObserver) {
        context.setProtocol(Protocol.grpc);
        context.putData(Dict.RESPONSE_STREAM_OBSERVER, responseObserver);
        context.putData(Dict.INPUT_DATA,inbound);
        OsxContext.pushThreadLocalContext(context);
        Osx.TransportOutbound result = null;
        try {
            ServiceRegisterInfo serviceRegisterInfo = this.serviceRegisterManager.getServiceWithLoadBalance(context,selfPartyIds.get(0),UriConstants.PEEK,false);
            Object serviceAdaptorObject = serviceRegisterInfo.getServiceAdaptor();
            ServiceAdaptorNew   serviceAdaptor = (ServiceAdaptorNew) serviceAdaptorObject;
            ConsumeRequest  consumeRequest =  new ConsumeRequest();
            consumeRequest.setTopic(inbound.getTopic());
            ConsumerResponse consumerResponse = (ConsumerResponse)serviceAdaptor.service(context,inbound );
            if(consumerResponse !=null){
                if(consumerResponse.isNeedRedirect()){
                    result = TransferUtil.redirectPeek(context,consumerResponse.getRedirectRouterInfo(),inbound);
                }else {
                    System.err.println("==========="+consumerResponse);
                    result = consumerResponse.toTransportOutbound();
                }
            }
        } catch (Exception e) {
            ExceptionInfo exceptionInfo = ErrorMessageUtil.handleExceptionExceptionInfo(context, e);
            context.setReturnCode(exceptionInfo.getCode());
            context.setReturnMsg(exceptionInfo.getMessage());
            result = Osx.TransportOutbound.newBuilder().setCode(exceptionInfo.getCode()).setMessage(exceptionInfo.getMessage()).build();
        }
        finally {
            FlowLogUtil.printFlowLog(context);
            OsxContext.popThreadLocalContext();
            OsxContext.release();
        }
        if (result != null) {
            responseObserver.onNext(result);
            responseObserver.onCompleted();
        }
    }

    @Override
    public void processGrpcPush(OsxContext  context ,Osx.PushInbound inbound, StreamObserver<Osx.TransportOutbound> responseObserver) {
        Osx.TransportOutbound result = null;
        OsxContext.pushThreadLocalContext(context);
        String desNodeId  = context.getDesNodeId();
        String srcNodeId = context.getSrcNodeId();

        try {
            if(MetaInfo.PROPERTY_SELF_PARTY.contains(desNodeId)) {
                    ServiceRegisterInfo serviceRegisterInfo = this.serviceRegisterManager.getServiceWithLoadBalance(context, selfPartyIds.get(0), UriConstants.PUSH, false);
                    AbstractServiceAdaptorNew serviceAdaptor = serviceRegisterInfo.getServiceAdaptor();
                    ProduceRequest produceRequest = new ProduceRequest();
                    produceRequest.setPayload(inbound.getPayload().toByteArray());
                    produceRequest.setTopic(inbound.getTopic());
                    ProduceResponse produceResponse = (ProduceResponse) serviceAdaptor.service(context, produceRequest);
                    if (produceResponse != null) {
                        result = Osx.TransportOutbound.newBuilder().setCode(produceResponse.getCode()).setMessage(produceResponse.getMsg()).build();
                    }
            }else{
                    RouterInfo  routerInfo = routerService.route(srcNodeId,Dict.DEFAULT,desNodeId,Dict.DEFAULT);
                    context.setActionType(MSG_REDIRECT.name());
                    Osx.Inbound.Builder   inboundBuilder =  Osx.Inbound.newBuilder();
                    inboundBuilder.setPayload(inbound.toByteString());
                    Osx.Outbound  outbound = TransferUtil.redirect(context, inboundBuilder.build(), routerInfo, true);
                    Osx.TransportOutbound.Builder   transportOutboundBuilder = Osx.TransportOutbound.newBuilder();
                    context.setReturnCode(outbound.getCode());
                    context.setReturnMsg(outbound.getMessage());
                    transportOutboundBuilder.setCode(outbound.getCode());
                    transportOutboundBuilder.setMessage(outbound.getMessage());
                    result  = transportOutboundBuilder.build();
                }
        } catch (Exception e) {
            ExceptionInfo exceptionInfo = ErrorMessageUtil.handleExceptionExceptionInfo(context, e);
            //this.writeHttpRespose(response, exceptionInfo.getCode(),exceptionInfo.getMessage(),null);
            context.setReturnCode(exceptionInfo.getCode());
            context.setReturnMsg(exceptionInfo.getMessage());
            result = Osx.TransportOutbound.newBuilder().setCode(exceptionInfo.getCode()).setMessage(exceptionInfo.getMessage()).build();
        } finally {
            FlowLogUtil.printFlowLog(context);
            OsxContext.popThreadLocalContext();
            OsxContext.release();
        }
        if (result != null) {
            responseObserver.onNext(result);
            responseObserver.onCompleted();
        }
    }
    //只有集群内部可以访问
    @Override
    public void processGrpcPop(OsxContext  context ,Osx.PopInbound inbound, StreamObserver<Osx.TransportOutbound> responseObserver) {

        context.setProtocol(Protocol.grpc);
        context.putData(Dict.RESPONSE_STREAM_OBSERVER, responseObserver);
        context.putData(Dict.INPUT_DATA,inbound);
        OsxContext.pushThreadLocalContext(context);
        Osx.TransportOutbound result = null;
        try {
            ServiceRegisterInfo serviceRegisterInfo = this.serviceRegisterManager.getServiceWithLoadBalance(context,selfPartyIds.get(0),UriConstants.POP,false);
            Object serviceAdaptorObject = serviceRegisterInfo.getServiceAdaptor();
            ServiceAdaptorNew   serviceAdaptor = (ServiceAdaptorNew) serviceAdaptorObject;
            ConsumeRequest  consumeRequest = new ConsumeRequest();
            consumeRequest.setTopic(inbound.getTopic());
            consumeRequest.setNeedBlock(true);
            consumeRequest.setTimeout(inbound.getTimeout());

            ConsumerResponse consumerResponse = (ConsumerResponse)serviceAdaptor.service(context,consumeRequest );
            if(consumerResponse !=null){
                if(consumerResponse.isNeedRedirect()){
                    result = TransferUtil.redirectPop(context,consumerResponse.getRedirectRouterInfo(),inbound);
                }else {
                    result = consumerResponse.toTransportOutbound();
                }
            }
        } catch (Exception e) {
            ExceptionInfo exceptionInfo = ErrorMessageUtil.handleExceptionExceptionInfo(context, e);
            context.setReturnCode(exceptionInfo.getCode());
            context.setReturnMsg(exceptionInfo.getMessage());
            result = Osx.TransportOutbound.newBuilder().setCode(exceptionInfo.getCode()).setMessage(exceptionInfo.getMessage()).build();
        }
        finally {
            FlowLogUtil.printFlowLog(context);
            OsxContext.popThreadLocalContext();
            OsxContext.release();
        }
        if (result != null) {
            responseObserver.onNext(result);
            responseObserver.onCompleted();
        }
    }
    //只有集群内部访问
    @Override
    public void processGrpcRelease(OsxContext  context , Osx.ReleaseInbound inbound, StreamObserver<Osx.TransportOutbound> responseObserver) {

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




    public  byte[] read(InputStream input) throws IOException {

            byte[] result = null;
            byte[] split = new byte[1024];
            int length=0;
            int count;
            while( (count = input.read(split))!=-1){
                byte[] temp =new  byte[length+count];
                System.arraycopy(split, 0, temp, length, count);
                if(result!=null) {
                    System.arraycopy(result, 0,temp,0,length );
                }
                result =  temp;
                length =  result.length;
            }
            return result;
    }


}
