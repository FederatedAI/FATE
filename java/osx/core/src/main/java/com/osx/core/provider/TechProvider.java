package com.osx.core.provider;

import io.grpc.stub.StreamObserver;
import org.ppc.ptp.Osx;


import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

public interface TechProvider {

    void processHttpInvoke(HttpServletRequest  httpServletRequest,HttpServletResponse httpServletResponse);

    void processGrpcInvoke(Osx.Inbound request,
                           io.grpc.stub.StreamObserver<Osx.Outbound> responseObserver);

    String getProviderId();

    public StreamObserver<Osx.Inbound> processGrpcTransport(Osx.Inbound inbound, io.grpc.stub.StreamObserver<Osx.Outbound> responseObserver);

}
