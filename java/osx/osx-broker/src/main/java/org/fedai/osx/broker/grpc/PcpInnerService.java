package org.fedai.osx.broker.grpc;

import com.google.inject.Inject;
import com.google.inject.Singleton;
import lombok.extern.slf4j.Slf4j;
import org.fedai.osx.broker.provider.TechProviderRegister;
import org.fedai.osx.broker.util.ContextUtil;
import org.fedai.osx.core.constant.UriConstants;
import org.fedai.osx.core.context.OsxContext;
import org.fedai.osx.core.provider.TechProvider;
import org.ppc.ptp.PrivateTransferProtocolGrpc;
import org.ppc.ptp.PrivateTransferTransportGrpc;
@Singleton
@Slf4j
public class PcpInnerService extends PrivateTransferTransportGrpc.PrivateTransferTransportImplBase {

    @Inject
    TechProviderRegister techProviderRegister;

    public void peek(org.ppc.ptp.Osx.PeekInbound request,
                     io.grpc.stub.StreamObserver<org.ppc.ptp.Osx.TransportOutbound> responseObserver) {
        OsxContext osxContext = new OsxContext();
        osxContext.setUri(UriConstants.PEEK);
        ContextUtil.assableContextFromInbound(osxContext);
        TechProvider techProvider= techProviderRegister.getTechProvider(osxContext);
        techProvider.processGrpcPeek(osxContext,request, responseObserver);
    }

    public void pop(org.ppc.ptp.Osx.PopInbound request,
                    io.grpc.stub.StreamObserver<org.ppc.ptp.Osx.TransportOutbound> responseObserver) {
        OsxContext osxContext = new OsxContext();
        osxContext.setUri(UriConstants.POP);
        ContextUtil.assableContextFromInbound(osxContext);
        TechProvider techProvider=techProviderRegister.getTechProvider(osxContext);
        techProvider.processGrpcPop(osxContext,request, responseObserver);
    }

    public void push(org.ppc.ptp.Osx.PushInbound request,
                     io.grpc.stub.StreamObserver<org.ppc.ptp.Osx.TransportOutbound> responseObserver) {
        OsxContext osxContext = new OsxContext();
        osxContext.setUri(UriConstants.PUSH);
        ContextUtil.assableContextFromInbound(osxContext);
        TechProvider techProvider= techProviderRegister.getTechProvider(osxContext);
        techProvider.processGrpcPush(osxContext,request, responseObserver);
    }

    public void release(org.ppc.ptp.Osx.ReleaseInbound request,
                        io.grpc.stub.StreamObserver<org.ppc.ptp.Osx.TransportOutbound> responseObserver) {
        OsxContext osxContext = new OsxContext();
        osxContext.setUri(UriConstants.RELEASE);
        ContextUtil.assableContextFromInbound(osxContext);
        TechProvider techProvider= techProviderRegister.getTechProvider(osxContext);
        techProvider.processGrpcRelease(osxContext,request, responseObserver);
    }

}
