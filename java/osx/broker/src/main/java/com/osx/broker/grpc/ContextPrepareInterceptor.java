package com.osx.broker.grpc;

import io.grpc.*;

public class ContextPrepareInterceptor implements ServerInterceptor {

    public static Context.Key<Object> sourceIp = Context.key("sourceIp");

    @Override
    public <ReqT, RespT> ServerCall.Listener<ReqT> interceptCall(ServerCall<ReqT, RespT> call, Metadata headers, ServerCallHandler<ReqT, RespT> next) {
        String remoteAddr = call.getAttributes().get(Grpc.TRANSPORT_ATTR_REMOTE_ADDR).toString();
        String[] remoteAddrSplited = remoteAddr.split(":");
        String remoteIp = remoteAddrSplited[0].replaceAll("\\/", "");
        Context context = Context.current().withValue(sourceIp, remoteIp);
        return Contexts.interceptCall(context, call, headers, next);
    }

}