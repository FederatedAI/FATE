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
import io.grpc.stub.StreamObserver;
import org.fedai.osx.broker.constants.ServiceType;
import org.fedai.osx.broker.pojo.*;
import org.fedai.osx.broker.router.DefaultFateRouterServiceImpl;
import org.fedai.osx.broker.service.ServiceRegisterInfo;
import org.fedai.osx.broker.service.ServiceRegisterManager;
import org.fedai.osx.broker.util.TransferUtil;
import org.fedai.osx.core.config.MetaInfo;
import org.fedai.osx.core.constant.*;
import org.fedai.osx.core.context.OsxContext;
import org.fedai.osx.core.context.Protocol;
import org.fedai.osx.core.exceptions.ErrorMessageUtil;
import org.fedai.osx.core.exceptions.ExceptionInfo;
import org.fedai.osx.core.exceptions.InvalidUriException;
import org.fedai.osx.core.exceptions.NoRouterInfoException;
import org.fedai.osx.core.provider.TechProvider;
import org.fedai.osx.core.router.RouterInfo;
import org.fedai.osx.core.service.AbstractServiceAdaptorNew;
import org.fedai.osx.core.service.ServiceAdaptorNew;
import org.fedai.osx.core.utils.FlowLogUtil;
import org.fedai.osx.core.utils.JsonUtil;
import org.ppc.ptp.Osx;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.servlet.AsyncContext;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.util.Base64;
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
    Base64.Encoder base64Encoder = Base64.getEncoder();
    Base64.Decoder base64Deonder = Base64.getDecoder();

    public FateTechProvider() {
        selfPartyIds = Lists.newArrayList(MetaInfo.PROPERTY_SELF_PARTY);
    }

    @Override
    public void processHttpInvoke(OsxContext osxContext, HttpServletRequest request, HttpServletResponse response) {

        try {
            OsxContext.pushThreadLocalContext(osxContext);
            osxContext.setProtocol(Protocol.http);
            osxContext.putData(Dict.HTTP_SERVLET_RESPONSE, response);
            byte[] reqBody = TransferUtil.read(request.getInputStream());
            System.err.println("receive======" + new String(reqBody));
            HttpInvoke httpInvoke = JsonUtil.json2Object(reqBody, HttpInvoke.class);
//            System.err.println("req map ==="+ reqMap);
//            byte[]  oriPayload = base64Deonder.decode(reqMap.get("payload").toString());
//            System.err.println("oriPayload : "+new String (oriPayload));
//            HttpInvoke  httpInvoke = new  HttpInvoke();
//            httpInvoke.setData(oriPayload);
//            System.err.println("ori payload ==="+new String(base64Deonder.decode(oriPayload)));
            //inboundBuilder.setPayload(ByteString.copyFrom(payload));
            HttpInvokeResult httpInvokeResult = (HttpInvokeResult) this.handleInner(osxContext, httpInvoke);
            response.setContentType(Dict.CONTENT_TYPE_JSON_UTF8);
            TransferUtil.writeHttpRespose(response, httpInvokeResult.getCode(), httpInvokeResult.getMessage(), JsonUtil.object2Json(httpInvokeResult).getBytes(StandardCharsets.UTF_8));
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            OsxContext.popThreadLocalContext();
            OsxContext.release();
        }
    }


    @Override
    public void processGrpcInvoke(OsxContext context, Osx.Inbound request, StreamObserver<Osx.Outbound> responseObserver) {
        try {
            context.setProtocol(Protocol.grpc);
            context.putData(Dict.RESPONSE_STREAM_OBSERVER, responseObserver);
            OsxContext.pushThreadLocalContext(context);
            Osx.Outbound result = (Osx.Outbound) handleInner(context, request);
            if (result != null) {
                responseObserver.onNext(result);
                responseObserver.onCompleted();
            }
        } finally {
            OsxContext.popThreadLocalContext();
            OsxContext.release();
        }

    }

    @Override
    public void processHttpPeek(OsxContext context, HttpServletRequest httpServletRequest, HttpServletResponse httpServletResponse) {
        context.setProtocol(Protocol.http);
        try {
            context.putData(Dict.HTTP_SERVLET_RESPONSE, httpServletResponse);
            //     OsxContext.pushThreadLocalContext(context);
            byte[] payload = TransferUtil.read(httpServletRequest.getInputStream());
            ConsumeRequest consumeRequest = JsonUtil.json2Object(new String(payload), ConsumeRequest.class);
            ServiceRegisterInfo serviceRegisterInfo = this.serviceRegisterManager.getServiceWithLoadBalance(context, selfPartyIds.get(0), UriConstants.PEEK, false);
            Object serviceAdaptorObject = serviceRegisterInfo.getServiceAdaptor();
            ServiceAdaptorNew serviceAdaptor = (ServiceAdaptorNew) serviceAdaptorObject;
            ConsumerResponse consumerResponse = (ConsumerResponse) serviceAdaptor.service(context, consumeRequest);
            httpServletResponse.setContentType(Dict.CONTENT_TYPE_JSON_UTF8);
            TransferUtil.writeHttpRespose(httpServletResponse, consumerResponse.getCode(), consumerResponse.getMsg(), JsonUtil.object2Json(consumerResponse).getBytes(StandardCharsets.UTF_8));
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
//            OsxContext.popThreadLocalContext();
//            OsxContext.release();
            FlowLogUtil.printFlowLog(context);
        }
    }

    @Override
    public void processHttpPush(OsxContext context, HttpServletRequest httpServletRequest, HttpServletResponse httpServletResponse) {
        context.setProtocol(Protocol.http);
        try {
            String desNodeId = context.getDesNodeId();
            byte[] body = TransferUtil.read(httpServletRequest.getInputStream());

            if (MetaInfo.PROPERTY_SELF_PARTY.contains(desNodeId)) {
                context.putData(Dict.HTTP_SERVLET_RESPONSE, httpServletResponse);
                //OsxContext.pushThreadLocalContext(context);

                ProduceRequest produceRequest = JsonUtil.json2Object(new String(body), ProduceRequest.class);
//                    ProduceRequest produceRequest = new ProduceRequest();
//                    produceRequest.setTopic(produceData.get(Dict.TOPIC) != null ? produceData.get(Dict.TOPIC).toString() : "");

                //  byte[] oriData = base64Deonder.decode(produceData.get(Dict.PAYLOAD) != null ? produceData.get(Dict.PAYLOAD).toString() : null);
                //  System.err.println("receive ======" + new String(oriData));
                //  produceRequest.setPayload(oriData);
                ServiceRegisterInfo serviceRegisterInfo = this.serviceRegisterManager.getServiceWithLoadBalance(context, selfPartyIds.get(0), UriConstants.PUSH, false);
                Object serviceAdaptorObject = serviceRegisterInfo.getServiceAdaptor();
                ServiceAdaptorNew serviceAdaptor = (ServiceAdaptorNew) serviceAdaptorObject;
                ProduceResponse produceResponse = (ProduceResponse) serviceAdaptor.service(context, produceRequest);
                httpServletResponse.setContentType(Dict.CONTENT_TYPE_JSON_UTF8);
                TransferUtil.writeHttpRespose(httpServletResponse, produceResponse.getCode(), produceResponse.getMsg(), JsonUtil.object2Json(produceResponse).getBytes(StandardCharsets.UTF_8));
            } else {
                HttpInvoke httpInvoke = new HttpInvoke();
                httpInvoke.setPayload(body);
                RouterInfo routerInfo = routerService.route(context.getSrcNodeId(), "", context.getDesNodeId(), "");
                context.setRouterInfo(routerInfo);
                context.setActionType(ActionType.MSG_REDIRECT.name());

                TransferUtil.redirect(context, httpInvoke, routerInfo, true);
            }
            // TODO: 2023/11/6   这里需要对异常情况进行处理
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
//            OsxContext.popThreadLocalContext();
//            OsxContext.release();
            FlowLogUtil.printFlowLog(context);
        }
    }

    @Override
    public void processHttpPop(OsxContext context, HttpServletRequest httpServletRequest, HttpServletResponse httpServletResponse) {
        context.setProtocol(Protocol.http);
        try {
            final AsyncContext ctxt = httpServletRequest.startAsync();
            ctxt.setTimeout(Integer.MAX_VALUE);
            context.putData(Dict.HTTP_ASYNC_CONTEXT, ctxt);
            ctxt.start(new Runnable() {
                @Override
                public void run() {
                    try {
                        OsxContext.pushThreadLocalContext(context);
                        byte[] payload = TransferUtil.read(ctxt.getRequest().getInputStream());
                        ConsumeRequest consumeRequest = JsonUtil.json2Object(new String(payload), ConsumeRequest.class);
                        consumeRequest.setNeedBlock(true);
                        ServiceRegisterInfo serviceRegisterInfo = serviceRegisterManager.getServiceWithLoadBalance(context, selfPartyIds.get(0), UriConstants.POP, false);
                        Object serviceAdaptorObject = serviceRegisterInfo.getServiceAdaptor();
                        ServiceAdaptorNew serviceAdaptor = (ServiceAdaptorNew) serviceAdaptorObject;

                        ConsumerResponse consumerResponse = (ConsumerResponse) serviceAdaptor.service(context, consumeRequest);
                        logger.info("consumer result ==================== {}", consumerResponse);
                        if (consumerResponse != null) {
                            if (!StatusCode.CONSUME_NO_MESSAGE.equals(consumerResponse.getCode())) {
                                httpServletResponse.setContentType(Dict.CONTENT_TYPE_JSON_UTF8);
                                byte[] respContent = consumerResponse.getPayload();
                                if (respContent != null) {
                                    consumerResponse.setPayload(base64Encoder.encode(respContent));
                                }
                                TransferUtil.writeHttpRespose(ctxt.getResponse(), consumerResponse.getCode(),
                                        consumerResponse.getMsg(), JsonUtil.object2Json(consumerResponse).getBytes(StandardCharsets.UTF_8));
                                ctxt.complete();
                            } else {


                            }
                        }
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }
            });


        } catch (Exception e) {
            e.printStackTrace();
            ExceptionInfo exceptionInfo = this.handleExceptionInfo(context, e);
        } finally {
            OsxContext.popThreadLocalContext();
            OsxContext.release();
            FlowLogUtil.printFlowLog(context);
        }
    }

    @Override
    public void processHttpRelease(OsxContext context, HttpServletRequest httpServletRequest, HttpServletResponse httpServletResponse) {
        context.setProtocol(Protocol.http);
        try {
            context.putData(Dict.HTTP_SERVLET_RESPONSE, httpServletResponse);
            OsxContext.pushThreadLocalContext(context);
            byte[] payload = TransferUtil.read(httpServletRequest.getInputStream());
            ReleaseRequest releaseRequest = JsonUtil.json2Object(new String(payload), ReleaseRequest.class);

            ServiceRegisterInfo serviceRegisterInfo = this.serviceRegisterManager.getServiceWithLoadBalance(context, selfPartyIds.get(0), UriConstants.RELEASE, false);
            Object serviceAdaptorObject = serviceRegisterInfo.getServiceAdaptor();
            ServiceAdaptorNew serviceAdaptor = (ServiceAdaptorNew) serviceAdaptorObject;
            ReleaseResponse response = (ReleaseResponse) serviceAdaptor.service(context, releaseRequest);
            TransferUtil.writeHttpRespose(httpServletResponse, response.getCode(), response.getMessage(), JsonUtil.object2Json(response).getBytes(StandardCharsets.UTF_8));
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            OsxContext.popThreadLocalContext();
            OsxContext.release();
            FlowLogUtil.printFlowLog(context);
        }
    }


    private Object handleInner(OsxContext context, Object request) {
        Object result = null;
        try {
            String uri = context.getUri();
            String instId = context.getDesInstId();
            String nodeId = context.getDesNodeId();
            if (MetaInfo.PROPERTY_SELF_PARTY.contains(nodeId)) {
                ServiceRegisterInfo serviceRegisterInfo = this.serviceRegisterManager.getServiceWithLoadBalance(context, nodeId, uri, true);
                if (serviceRegisterInfo != null) {
                    ServiceType serviceType = serviceRegisterInfo.getServiceType();
                    switch (serviceType) {
                        case inner:
                            Object serviceAdaptorObject = serviceRegisterInfo.getServiceAdaptor();
                            ServiceAdaptorNew serviceAdaptor = (ServiceAdaptorNew) serviceAdaptorObject;
                            Object requestObj = serviceAdaptor.decode(request);
                            Object responseObj = serviceAdaptor.service(context, requestObj);
                            result = serviceAdaptor.toOutbound(responseObj);
                            break;
//                        case  rpc:
//                            RouterInfo routerInfo = serviceRegisterInfo.getRouterInfo();
//                            result = TransferUtil.redirect(context,request,routerInfo,true);
//                            break;
                    }
                } else {
                    logger.error("invalid  uri {}", uri);
                    throw new InvalidUriException();
                }
            } else {
                //外部则转发
                // RouterService routerService = routerRegister.getRouterService(MetaInfo.PROPERTY_FATE_TECH_PROVIDER);

                RouterInfo routerInfo = routerService.routePtp(context.getSrcInstId(), context.getSrcNodeId(), context.getDesInstId(), context.getDesNodeId());
                if (routerInfo != null) {
                    // TODO: 2023/11/3   这里要根据request 类型不同做不同处理
                    result = (HttpInvokeResult) TransferUtil.redirect(context, (Osx.Inbound) request, routerInfo, true);
                } else {
                    logger.error("can not found router info {} {}", context.getDesInstId(), context.getDesNodeId());
                    throw new NoRouterInfoException("can not found router info");
                }
            }
        } catch (Exception e) {
            logger.error("", e);
            ExceptionInfo exceptionInfo = handleExceptionInfo(context, e);
            if (context.getProtocol().equals(Protocol.grpc)) {
                Osx.Outbound.Builder builder = Osx.Outbound.newBuilder();
                if (exceptionInfo.getCode() != null)
                    builder.setCode(exceptionInfo.getCode());
                result = builder.setMessage(exceptionInfo.getMessage()).build();
            } else {
                HttpInvokeResult httpInvokeResult = new HttpInvokeResult();
                if (exceptionInfo.getCode() != null)
                    httpInvokeResult.setCode(exceptionInfo.getCode());
                httpInvokeResult.setMessage(exceptionInfo.getMessage());
                result = httpInvokeResult;
            }
        } finally {
            FlowLogUtil.printFlowLog(context);
            OsxContext.popThreadLocalContext();
            OsxContext.release();
        }
        return result;
    }

    private ExceptionInfo handleExceptionInfo(OsxContext context, Throwable e) {
        ExceptionInfo exceptionInfo = ErrorMessageUtil.handleExceptionExceptionInfo(context, e);
        //this.writeHttpRespose(response, exceptionInfo.getCode(),exceptionInfo.getMessage(),null);
        context.setReturnCode(exceptionInfo.getCode());
        context.setReturnMsg(exceptionInfo.getMessage());
        return exceptionInfo;
    }


    @Override
    public void processGrpcPeek(OsxContext context, Osx.PeekInbound inbound, StreamObserver<Osx.TransportOutbound> responseObserver) {
        context.setProtocol(Protocol.grpc);
        context.putData(Dict.RESPONSE_STREAM_OBSERVER, responseObserver);
        context.putData(Dict.INPUT_DATA, inbound);
        OsxContext.pushThreadLocalContext(context);
        Osx.TransportOutbound result = null;
        try {
            ServiceRegisterInfo serviceRegisterInfo = this.serviceRegisterManager.getServiceWithLoadBalance(context, selfPartyIds.get(0), UriConstants.PEEK, false);
            Object serviceAdaptorObject = serviceRegisterInfo.getServiceAdaptor();
            ServiceAdaptorNew serviceAdaptor = (ServiceAdaptorNew) serviceAdaptorObject;
            ConsumeRequest consumeRequest = new ConsumeRequest();
            consumeRequest.setTopic(inbound.getTopic());
            ConsumerResponse consumerResponse = (ConsumerResponse) serviceAdaptor.service(context, consumeRequest);
            if (consumerResponse != null) {
                if (consumerResponse.isNeedRedirect()) {
                    result = TransferUtil.redirectPeek(context, consumerResponse.getRedirectRouterInfo(), inbound);
                } else {
                    result = consumerResponse.toTransportOutbound();
                }
            }
        } catch (Exception e) {
            ExceptionInfo exceptionInfo = this.handleExceptionInfo(context, e);
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

    @Override
    public void processGrpcPush(OsxContext context, Osx.PushInbound inbound, StreamObserver<Osx.TransportOutbound> responseObserver) {
        Osx.TransportOutbound result = null;
        context.setProtocol(Protocol.grpc);
        OsxContext.pushThreadLocalContext(context);
        String desNodeId = context.getDesNodeId();
        String srcNodeId = context.getSrcNodeId();

        try {
            if (MetaInfo.PROPERTY_SELF_PARTY.contains(desNodeId)) {
                ServiceRegisterInfo serviceRegisterInfo = this.serviceRegisterManager.getServiceWithLoadBalance(context, selfPartyIds.get(0), UriConstants.PUSH, false);
                AbstractServiceAdaptorNew serviceAdaptor = serviceRegisterInfo.getServiceAdaptor();
                ProduceRequest produceRequest = new ProduceRequest();
                produceRequest.setPayload(inbound.getPayload().toByteArray());
                produceRequest.setTopic(inbound.getTopic());
                ProduceResponse produceResponse = (ProduceResponse) serviceAdaptor.service(context, produceRequest);
                if (produceResponse != null) {
                    result = Osx.TransportOutbound.newBuilder().setCode(produceResponse.getCode()).setMessage(produceResponse.getMsg()).build();
                }
            } else {
                RouterInfo routerInfo = routerService.route(srcNodeId, Dict.DEFAULT, desNodeId, Dict.DEFAULT);
                context.setActionType(MSG_REDIRECT.name());
                Osx.Inbound.Builder inboundBuilder = Osx.Inbound.newBuilder();
                inboundBuilder.setPayload(inbound.toByteString());
                context.setUri(UriConstants.PUSH);
                context.setTopic(inbound.getTopic());
                context.setRouterInfo(routerInfo);
                Osx.Outbound outbound = (Osx.Outbound) TransferUtil.redirect(context, inboundBuilder.build(), routerInfo, true);
                Osx.TransportOutbound.Builder transportOutboundBuilder = Osx.TransportOutbound.newBuilder();
                context.setReturnCode(outbound.getCode());
                context.setReturnMsg(outbound.getMessage());
                transportOutboundBuilder.setCode(outbound.getCode());
                transportOutboundBuilder.setMessage(outbound.getMessage());
                result = transportOutboundBuilder.build();
            }
        } catch (Exception e) {
            ExceptionInfo exceptionInfo = this.handleExceptionInfo(context, e);
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
    public void processGrpcPop(OsxContext context, Osx.PopInbound inbound, StreamObserver<Osx.TransportOutbound> responseObserver) {
        context.setProtocol(Protocol.grpc);
        context.putData(Dict.RESPONSE_STREAM_OBSERVER, responseObserver);
        context.putData(Dict.INPUT_DATA, inbound);
        OsxContext.pushThreadLocalContext(context);
        Osx.TransportOutbound result = null;
        try {
            ServiceRegisterInfo serviceRegisterInfo = this.serviceRegisterManager.getServiceWithLoadBalance(context, selfPartyIds.get(0), UriConstants.POP, false);
            Object serviceAdaptorObject = serviceRegisterInfo.getServiceAdaptor();
            ServiceAdaptorNew serviceAdaptor = (ServiceAdaptorNew) serviceAdaptorObject;
            ConsumeRequest consumeRequest = new ConsumeRequest();
            consumeRequest.setTopic(inbound.getTopic());
            consumeRequest.setNeedBlock(true);
            consumeRequest.setTimeout(inbound.getTimeout());
            ConsumerResponse consumerResponse = (ConsumerResponse) serviceAdaptor.service(context, consumeRequest);
            if (consumerResponse != null) {
                if (consumerResponse.isNeedRedirect()) {
                    result = TransferUtil.redirectPop(context, consumerResponse.getRedirectRouterInfo(), inbound);
                } else {
                    result = consumerResponse.toTransportOutbound();
                }
            }
        } catch (Exception e) {
            ExceptionInfo exceptionInfo = this.handleExceptionInfo(context, e);
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

    //只有集群内部访问
    @Override
    public void processGrpcRelease(OsxContext context, Osx.ReleaseInbound inbound, StreamObserver<Osx.TransportOutbound> responseObserver) {
        context.setProtocol(Protocol.grpc);
        context.putData(Dict.RESPONSE_STREAM_OBSERVER, responseObserver);
        context.putData(Dict.INPUT_DATA, inbound);
        OsxContext.pushThreadLocalContext(context);
        Osx.TransportOutbound result = null;
        try {
            ServiceRegisterInfo serviceRegisterInfo = this.serviceRegisterManager.getServiceWithLoadBalance(context, selfPartyIds.get(0), UriConstants.RELEASE, false);
            Object serviceAdaptorObject = serviceRegisterInfo.getServiceAdaptor();
            ServiceAdaptorNew serviceAdaptor = (ServiceAdaptorNew) serviceAdaptorObject;
            ReleaseRequest releaseRequest = new ReleaseRequest();
            releaseRequest.setTopic(inbound.getTopic());


            ReleaseResponse consumerResponse = (ReleaseResponse) serviceAdaptor.service(context, releaseRequest);
//            if(consumerResponse !=null){
//                if(consumerResponse.isNeedRedirect()){
//                    result = TransferUtil.redirectPop(context,consumerResponse.getRedirectRouterInfo(),inbound);
//                }else {
//                    result = consumerResponse.toTransportOutbound();
//                }
//            }
            result = Osx.TransportOutbound.newBuilder().setCode(consumerResponse.getCode()).setMessage(consumerResponse.getMessage()).build();


        } catch (Exception e) {
            ExceptionInfo exceptionInfo = this.handleExceptionInfo(context, e);
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


//    public  byte[] read(InputStream input) throws IOException {
//
//            byte[] result = null;
//            byte[] split = new byte[1024];
//            int length=0;
//            int count;
//            while( (count = input.read(split))!=-1){
//                byte[] temp =new  byte[length+count];
//                System.arraycopy(split, 0, temp, length, count);
//                if(result!=null) {
//                    System.arraycopy(result, 0,temp,0,length );
//                }
//                result =  temp;
//                length =  result.length;
//            }
//            return result;
//    }


}
