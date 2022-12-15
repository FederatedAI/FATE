package com.osx.broker.grpc;

import io.grpc.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class ServiceExceptionHandler implements ServerInterceptor {
    private static final Logger logger = LoggerFactory.getLogger(ServiceExceptionHandler.class);

    @Override
    public <ReqT, RespT> ServerCall.Listener<ReqT> interceptCall(ServerCall<ReqT, RespT> call,
                                                                 Metadata requestHeaders, ServerCallHandler<ReqT, RespT> next) {
        ServerCall.Listener<ReqT> delegate = next.startCall(call, requestHeaders);
        return new ForwardingServerCallListener.SimpleForwardingServerCallListener<ReqT>(delegate) {
            @Override
            public void onHalfClose() {
                try {
                    super.onHalfClose();
                } catch (Exception e) {
                    logger.error("ServiceException:", e);
                    call.close(Status.INTERNAL
                            .withCause(e)
                            .withDescription(e.getMessage()), new Metadata());
                }
            }
        };
    }
}