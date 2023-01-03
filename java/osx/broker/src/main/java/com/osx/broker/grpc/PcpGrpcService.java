package com.osx.broker.grpc;

import com.osx.broker.ServiceContainer;
import com.osx.core.exceptions.SysException;
import com.osx.core.provider.TechProvider;
import io.grpc.stub.StreamObserver;
import org.ppc.ptp.Osx;
import org.ppc.ptp.PrivateTransferProtocolGrpc;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Map;


public class PcpGrpcService extends PrivateTransferProtocolGrpc.PrivateTransferProtocolImplBase {

    /**
     * 流式接口
     *
     * @param responseObserver
     * @return
     */
    public io.grpc.stub.StreamObserver<Osx.Inbound> transport(
            io.grpc.stub.StreamObserver<Osx.Outbound> responseObserver) {
        return new PcpStreamObserver(responseObserver);
    }

    /**
     * 同步接口
     *
     * @param request
     * @param responseObserver
     */
    public void invoke(Osx.Inbound request,
                       io.grpc.stub.StreamObserver<Osx.Outbound> responseObserver) {


        Map<String, String> metaDataMap = request.getMetadataMap();
       // String version = metaDataMap.get(Pcp.Header.Version.name());
        String techProviderCode = metaDataMap.get(Osx.Header.TechProviderCode.name());

        TechProvider techProvider = ServiceContainer.techProviderRegister.select(techProviderCode);
        if (techProvider != null) {
            techProvider.processGrpcInvoke(request, responseObserver);
        }
    }

    public class PcpStreamObserver implements io.grpc.stub.StreamObserver<Osx.Inbound> {

        Logger logger = LoggerFactory.getLogger(PcpStreamObserver.class);
        boolean inited = false;
        TechProvider techProvider;
        StreamObserver<Osx.Outbound> responseObserver;
        StreamObserver<Osx.Inbound> requestObserver;
        public PcpStreamObserver(StreamObserver<Osx.Outbound> responseObserver) {
            this.responseObserver = responseObserver;
        }

        private void init(Osx.Inbound inbound) {

            Map<String, String> metaDataMap = inbound.getMetadataMap();
            // String version = metaDataMap.get(Pcp.Header.Version.name());

            logger.info("PcpStreamObserver init {}",metaDataMap);
           // System.err.println("pppppppppppppppppppppp "+metaDataMap);
            String techProviderCode = metaDataMap.get(Osx.Header.TechProviderCode.name());
            techProvider = ServiceContainer.techProviderRegister.select(techProviderCode);
            if (techProvider != null) {
                requestObserver = techProvider.processGrpcTransport(inbound, responseObserver);
            } else {
                //抛出异常
                logger.error("can not found TechProvider of {}",techProviderCode);
                throw  new SysException("invalid TechProviderCode "+techProviderCode);
            }
            inited = true;
            logger.info("PcpStreamObserver init over");
        }

        @Override
        public void onNext(Osx.Inbound inbound) {
            if (!inited) {
                init(inbound);
            }
            if (requestObserver != null) {
                requestObserver.onNext(inbound);
            } else {
                throw new RuntimeException();
            }
        }

        @Override
        public void onError(Throwable throwable) {
            if (requestObserver != null) {
                requestObserver.onError(throwable);
            } else {
                throw new RuntimeException();
            }
        }

        @Override
        public void onCompleted() {
            if (requestObserver != null) {
                requestObserver.onCompleted();
            } else {
                throw new RuntimeException();
            }
        }
    }


}
