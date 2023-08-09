//package  org.fedai.osx.broker.interceptor;
//
//import  org.fedai.osx.broker.grpc.PushRequestDataWrap;
//import  org.fedai.osx.core.context.Context;
//import  org.fedai.osx.core.service.InboundPackage;
//import  org.fedai.osx.core.service.Interceptor;
//import com.webank.ai.eggroll.api.networking.proxy.Proxy;
//
//import static  org.fedai.osx.broker.util.TransferUtil.assableContextFromProxyPacket;
//
//public class PushHandleInterceptor implements Interceptor<PushRequestDataWrap,Object> {
//
//     public void doPreProcess(Context context, InboundPackage<PushRequestDataWrap> inboundPackage) throws Exception {
//            PushRequestDataWrap pushRequestDataWrap =inboundPackage.getBody();
//            Proxy.Packet packet = pushRequestDataWrap.getPacket();
////            assableContextFromProxyPacket(context ,packet);
//    }
//
//}
