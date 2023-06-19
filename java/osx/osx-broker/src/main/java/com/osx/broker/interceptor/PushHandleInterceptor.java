//package com.osx.broker.interceptor;
//
//import com.osx.broker.grpc.PushRequestDataWrap;
//import com.osx.core.context.Context;
//import com.osx.core.service.InboundPackage;
//import com.osx.core.service.Interceptor;
//import com.webank.ai.eggroll.api.networking.proxy.Proxy;
//
//import static com.osx.broker.util.TransferUtil.assableContextFromProxyPacket;
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
