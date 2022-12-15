package com.osx.tech.provider;


import com.google.common.base.Preconditions;
import com.osx.core.frame.Lifecycle;
import com.osx.core.context.Context;
import com.osx.core.provider.TechProvider;
import com.osx.core.ptp.TargetMethod;
import com.osx.core.service.InboundPackage;
import com.osx.core.service.OutboundPackage;
import com.osx.core.service.ServiceAdaptor;
import com.osx.broker.ServiceContainer;
import com.osx.broker.grpc.ContextUtil;
import com.osx.broker.interceptor.RequestHandleInterceptor;
import com.osx.broker.ptp.*;

import com.osx.broker.service.*;
import io.grpc.stub.StreamObserver;
import org.ppc.ptp.Pcp;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;

/**
 * FATE 相关实现
 */

public class FateTechProvider implements TechProvider, Lifecycle {

    ConcurrentMap<String, ServiceAdaptor>  serviceAdaptorConcurrentMap = new ConcurrentHashMap<>();

    RequestHandleInterceptor requestHandleInterceptor  ;

    @Override
    public void processInvoke(Pcp.Inbound request, StreamObserver<Pcp.Outbound> responseObserver) {
        Map<String, String> metaDataMap = request.getMetadataMap();

        String targetMethod = metaDataMap.get(Pcp.Metadata.forNumber(4).name());
        Context context = ContextUtil.buildContext();

        //RouterInfo routerInfo = ServiceContainer.fateRouterService.route(sourcePartyId,sourceComponentName,targetPartyId,targetComponentName);
        //context.setRouterInfo(routerInfo);
        ServiceAdaptor  serviceAdaptor = this.getServiceAdaptor(targetMethod);
        InboundPackage inboundPackage =  new InboundPackage();
        inboundPackage.setBody(request);
        OutboundPackage<Pcp.Outbound>  outboundPackage = serviceAdaptor.service(context,inboundPackage);
        if(outboundPackage.getData()!=null) {
            responseObserver.onNext(outboundPackage.getData());
            responseObserver.onCompleted();
        }
    }

    @Override
    public String getProviderId() {
        return "FT";
    }



    @Override
    public StreamObserver<Pcp.Inbound> processTransport(Pcp.Inbound fristPackage, StreamObserver<Pcp.Outbound> responseObserver) {

        return null;
    }

    @Override
    public void init() {
        Preconditions.checkArgument(ServiceContainer.fateRouterService!=null);
        requestHandleInterceptor = new RequestHandleInterceptor(ServiceContainer.fateRouterService);
        registerServiceAdaptor();
    }


    public FateTechProvider setRouterService(){

        return this;
    }

    @Override
    public void start() {

    }

    @Override
    public void destroy() {

    }

    private ServiceAdaptor   getServiceAdaptor(String  name){
      return   this.serviceAdaptorConcurrentMap.get(name);
    }

    private  void  registerServiceAdaptor(){
        this.serviceAdaptorConcurrentMap.put(TargetMethod.UNARY_CALL.name(),  new UnaryCallService());
        this.serviceAdaptorConcurrentMap.put(TargetMethod.PRODUCE_MSG.name(),new PtpProduceService().addPreProcessor(requestHandleInterceptor));
        this.serviceAdaptorConcurrentMap.put(TargetMethod.ACK_MSG.name(),new PtpAckService().addPreProcessor(requestHandleInterceptor));
        this.serviceAdaptorConcurrentMap.put(TargetMethod.CONSUME_MSG.name(),new PtpConsumeService().addPreProcessor(requestHandleInterceptor));
        this.serviceAdaptorConcurrentMap.put(TargetMethod.QUERY_TOPIC.name(),new PtpQueryTransferQueueService().addPreProcessor(requestHandleInterceptor));
        this.serviceAdaptorConcurrentMap.put(TargetMethod.CANCEL_TOPIC.name(), new PtpCancelTransferService().addPreProcessor(requestHandleInterceptor));
        this.serviceAdaptorConcurrentMap.put(TargetMethod.PUSH.name(), new  PtpPushService());
    }
}
