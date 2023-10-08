//package org.fedai.osx.broker.provider;
//
//import com.google.inject.Singleton;
//import io.grpc.stub.StreamObserver;
//import org.fedai.osx.api.constants.Protocol;
//import org.fedai.osx.api.context.Context;
//import org.fedai.osx.broker.util.ContextUtil;
//import org.fedai.osx.core.constant.Dict;
//import org.fedai.osx.core.context.OsxContext;
//import org.fedai.osx.core.exceptions.ErrorMessageUtil;
//import org.fedai.osx.core.exceptions.ExceptionInfo;
//import org.fedai.osx.core.exceptions.ParameterException;
//import org.fedai.osx.core.provider.TechProvider;
//import org.fedai.osx.core.service.InboundPackage;
//import org.fedai.osx.core.service.OutboundPackage;
//import org.fedai.osx.core.service.ServiceAdaptor;
//import org.fedai.osx.core.utils.FlowLogUtil;
//import org.ppc.ptp.Osx;
//
//import javax.servlet.http.HttpServletRequest;
//import javax.servlet.http.HttpServletResponse;
//import java.util.Map;
//
//public class PtpTechProvider implements TechProvider {
//    @Override
//    public void processHttpInvoke(OsxContext context , HttpServletRequest httpServletRequest, HttpServletResponse httpServletResponse) {
//
//    }
//
//    @Override
//    public void processGrpcInvoke(OsxContext  context ,Osx.Inbound request, StreamObserver<Osx.Outbound> responseObserver) {
////        Context context = ContextUtil.buildFateContext(Protocol.grpc);
////        context.putData(Dict.RESPONSE_STREAM_OBSERVER, responseObserver);
////        Osx.Outbound result = null;
////        try {
////            Map<String, String> metaDataMap = request.getMetadataMap();
////            String targetMethod = metaDataMap.get(Osx.Metadata.TargetMethod.name());
////            ServiceAdaptor serviceAdaptor = this.getServiceAdaptor(targetMethod);
////            if (serviceAdaptor == null) {
////                throw new ParameterException("invalid target method " + targetMethod);
////            }
////            InboundPackage inboundPackage = new InboundPackage();
////            inboundPackage.setBody(request);
////            OutboundPackage<Osx.Outbound> outboundPackage = serviceAdaptor.service(context, inboundPackage);
////            if (outboundPackage.getData() != null) {
////                result = outboundPackage.getData();
////            }
////        } catch (Exception e) {
////            ExceptionInfo exceptionInfo = ErrorMessageUtil.handleExceptionExceptionInfo(context, e);
////            //this.writeHttpRespose(response, exceptionInfo.getCode(),exceptionInfo.getMessage(),null);
////            context.setReturnCode(exceptionInfo.getCode());
////            context.setReturnMsg(exceptionInfo.getMessage());
////            FlowLogUtil.printFlowLog(context);
////            result = Osx.Outbound.newBuilder().setCode(exceptionInfo.getCode()).setMessage(exceptionInfo.getMessage()).build();
////        }
////        if (result != null) {
////            responseObserver.onNext(result);
////            responseObserver.onCompleted();
////        }
//    }
//
//    @Override
//    public StreamObserver<Osx.Inbound> processGrpcTransport(OsxContext  context ,Osx.Inbound inbound, StreamObserver<Osx.Outbound> responseObserver) {
//        return null;
//    }
//
//    @Override
//    public void processGrpcPeek(OsxContext  context ,Osx.PeekInbound inbound, StreamObserver<Osx.TransportOutbound> responseObserver) {
//
//    }
//
//    @Override
//    public void processGrpcPush(OsxContext  context ,Osx.PushInbound inbound, StreamObserver<Osx.TransportOutbound> responseObserver) {
//
//    }
//
//    @Override
//    public void processGrpcPop(OsxContext  context ,Osx.PopInbound inbound, StreamObserver<Osx.TransportOutbound> responseObserver) {
//
//    }
//
//    @Override
//    public void processGrpcRelease(OsxContext  context ,Osx.ReleaseInbound inbound, StreamObserver<Osx.TransportOutbound> responseObserver) {
//
//    }
//
////            "/v1/interconn/chan/pop":
////             "/v1/interconn/chan/push":
////             "/v1/interconn/chan/peek":
////             "/v1/interconn/chan/release":
////             "/v1/interconn/net/weave":
//    			//"/org.ppc.ptp.PrivateTransferTransport/peek":
//                //"/org.ppc.ptp.PrivateTransferTransport/pop":
//                // "/org.ppc.ptp.PrivateTransferTransport/push":
//                //"/org.ppc.ptp.PrivateTransferTransport/release":
//
//
//
//
//
//}
