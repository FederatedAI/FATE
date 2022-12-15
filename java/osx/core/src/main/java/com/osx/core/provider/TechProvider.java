package com.osx.core.provider;
import io.grpc.stub.StreamObserver;
import org.ppc.ptp.Pcp;
import org.ppc.ptp.Pcp.Inbound;
public interface TechProvider {
        void  processInvoke(Inbound request,
                           io.grpc.stub.StreamObserver<Pcp.Outbound> responseObserver);
        String getProviderId();

       public StreamObserver<Inbound> processTransport(Pcp.Inbound  inbound,io.grpc.stub.StreamObserver<Pcp.Outbound> responseObserver);

}
