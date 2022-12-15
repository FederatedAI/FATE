package com.osx.core.ptp;

public enum TargetMethod {

//            this.serviceAdaptorConcurrentMap.put("UNARY_CALL",  new UnaryCallService());
//        this.serviceAdaptorConcurrentMap.put("PRODUCE_MSG",new PtpProduceService().addPreProcessor(requestHandleInterceptor));
//        this.serviceAdaptorConcurrentMap.put("ACK_MSG",new PtpAckService().addPreProcessor(requestHandleInterceptor));
//        this.serviceAdaptorConcurrentMap.put("CONSUME_MSG",new PtpConsumeService().addPreProcessor(requestHandleInterceptor));
//        this.serviceAdaptorConcurrentMap.put("QUERY_TOPIC",new PtpQueryTransferQueueService().addPreProcessor(requestHandleInterceptor));
//        this.serviceAdaptorConcurrentMap.put("CANCEL_TOPIC", new PtpCancelTransferService().addPreProcessor(requestHandleInterceptor));
    UNARY_CALL,
    PRODUCE_MSG,
    ACK_MSG,
    CONSUME_MSG,
    QUERY_TOPIC,
    CANCEL_TOPIC,
    PUSH


}
