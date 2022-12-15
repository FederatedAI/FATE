package com.osx.broker.grpc;
import com.osx.core.provider.TechProvider;
import com.osx.broker.ServiceContainer;
import io.grpc.stub.StreamObserver;
import org.ppc.ptp.PrivateTransferProtocolGrpc;
import org.ppc.ptp.Pcp;



public class PcpGrpcService extends PrivateTransferProtocolGrpc.PrivateTransferProtocolImplBase {



    /**
     * 流式接口
     * @param responseObserver
     * @return
     */
    public io.grpc.stub.StreamObserver<org.ppc.ptp.Pcp.Inbound> transport(
            io.grpc.stub.StreamObserver<org.ppc.ptp.Pcp.Outbound> responseObserver) {
       return new  PcpStreamObserver(responseObserver);
    }
    /**
     * 同步接口
     * @param request
     * @param responseObserver
     */
    public void invoke(org.ppc.ptp.Pcp.Inbound request,
                       io.grpc.stub.StreamObserver<org.ppc.ptp.Pcp.Outbound> responseObserver) {
        TechProvider techProvider = ServiceContainer.techProviderRegister.select(request);
        if(techProvider!=null) {
            techProvider.processInvoke(request, responseObserver);
        }
    }
    public  class  PcpStreamObserver implements  io.grpc.stub.StreamObserver<org.ppc.ptp.Pcp.Inbound>  {

        public  PcpStreamObserver(StreamObserver<Pcp.Outbound> responseObserver){
            this.responseObserver = responseObserver;
        }
        boolean  init = false;
        TechProvider techProvider;
        StreamObserver<Pcp.Outbound> responseObserver;
        StreamObserver<Pcp.Inbound> requestObserver;
        private  void  init(Pcp.Inbound inbound){
            techProvider = ServiceContainer.techProviderRegister.select(inbound);
            if(techProvider!=null) {
                requestObserver = techProvider.processTransport(inbound,responseObserver);
            }
            else {
                //抛出异常

            }
        }

        @Override
        public void onNext(Pcp.Inbound inbound) {
            if(!init) {
                init(inbound);
            }
            if(requestObserver!=null){
                requestObserver.onNext(inbound);
            }else{
                throw  new RuntimeException();
            }


        }

        @Override
        public void onError(Throwable throwable) {
            if(requestObserver!=null){
                requestObserver.onError(throwable);
            }else{
                throw  new RuntimeException();
            }
        }

        @Override
        public void onCompleted() {
            if(requestObserver!=null){
                requestObserver.onCompleted();
            }else{
                throw  new RuntimeException();
            }
        }
    }






}
