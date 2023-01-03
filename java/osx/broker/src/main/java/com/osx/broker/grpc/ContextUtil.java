package com.osx.broker.grpc;

import com.osx.core.context.Context;

import java.util.UUID;

public class ContextUtil {

    public static Context buildContext() {
        Context context = new Context();
        context.setSourceIp(ContextPrepareInterceptor.sourceIp.get() != null ? ContextPrepareInterceptor.sourceIp.get().toString() : "");
        context.setCaseId(UUID.randomUUID().toString());
        return context;
    }
}
