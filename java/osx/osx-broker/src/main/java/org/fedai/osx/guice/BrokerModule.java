package org.fedai.osx.guice;

import com.google.inject.AbstractModule;
import com.google.inject.matcher.Matcher;
import com.google.inject.matcher.Matchers;

public class BrokerModule extends AbstractModule {

//    @Provides
//    public  CuratorZookeeperClient createCuratorZookeeperClient() {
//        if (MetaInfo.isCluster()) {
//            ZkConfig zkConfig = new ZkConfig(MetaInfo.PROPERTY_ZK_URL, 5000);
//            return new CuratorZookeeperClient(zkConfig);
//        }
//        return null;
//    }


//    @Provides
//    PtpUnaryCallService  getPtpUnaryCallService(
//                                                TokenValidatorInterceptor tokenValidatorInterceptor,
//                                                RouterInterceptor  routerInterceptor
//    ){
//        PtpUnaryCallService   ptpUnaryCallService = new  PtpUnaryCallService();
//         ptpUnaryCallService
//                .addPreProcessor(tokenValidatorInterceptor)
//                .addPreProcessor(routerInterceptor);
//         return ptpUnaryCallService;
//    }

//    @Provides
//    PtpProduceService   getPtpProduceService(RouterInterceptor  routerInterceptor,
//            TransferQueueManager transferQueueManager,
//            TokenApplyService tokenApplyService,
//                                                 FlowCounterManager flowCounterManager){
//        PtpProduceService  ptpProduceService = new  PtpProduceService();
//        ptpProduceService.addPreProcessor(routerInterceptor);
//        ptpProduceService.setTokenApplyService(tokenApplyService);
//        ptpProduceService.setTransferQueueManager(transferQueueManager);
//        ptpProduceService.setFlowCounterManager(flowCounterManager);
//        return  ptpProduceService;
//    }


//    @Provides
//    PtpUnaryCallService  getPtpUnaryCallService(PcpHandleInterceptor requestHandleInterceptor,
//                                                TokenValidatorInterceptor tokenValidatorInterceptor,
//                                                RouterInterceptor  routerInterceptor
//    ){
//        PtpUnaryCallService   ptpUnaryCallService = new  PtpUnaryCallService();
//        ptpUnaryCallService .addPreProcessor(requestHandleInterceptor)
//                .addPreProcessor(tokenValidatorInterceptor)
//                .addPreProcessor(routerInterceptor);
//        return ptpUnaryCallService;
//    }


//    private void registerServiceAdaptor(PcpHandleInterceptor  ) {
//        this.serviceAdaptorConcurrentMap.put(TargetMethod.UNARY_CALL.name(), new PtpUnaryCallService()
//                .addPreProcessor(requestHandleInterceptor)
//                .addPreProcessor(tokenValidatorInterceptor)
//                .addPreProcessor(routerInterceptor));
//        this.serviceAdaptorConcurrentMap.put(TargetMethod.PRODUCE_MSG.name(), new PtpProduceService()
//                .addPreProcessor(requestHandleInterceptor)
//                .addPreProcessor(routerInterceptor));
//        this.serviceAdaptorConcurrentMap.put(TargetMethod.ACK_MSG.name(), new PtpAckService()
//                .addPreProcessor(requestHandleInterceptor));
//        this.serviceAdaptorConcurrentMap.put(TargetMethod.CONSUME_MSG.name(), new PtpConsumeService()
//                .addPreProcessor(requestHandleInterceptor));
//        this.serviceAdaptorConcurrentMap.put(TargetMethod.QUERY_TOPIC.name(), new PtpQueryTransferQueueService()
//                .addPreProcessor(requestHandleInterceptor));
//        this.serviceAdaptorConcurrentMap.put(TargetMethod.CANCEL_TOPIC.name(), new PtpCancelTransferService()
//                .addPreProcessor(requestHandleInterceptor));
//        this.serviceAdaptorConcurrentMap.put(TargetMethod.PUSH.name(), new PtpPushService());
//        this.serviceAdaptorConcurrentMap.put(TargetMethod.APPLY_TOKEN.name(), new PtpClusterTokenApplyService());
//        this.serviceAdaptorConcurrentMap.put(TargetMethod.APPLY_TOPIC.name(), new PtpClusterTopicApplyService());
//        // this.serviceAdaptorConcurrentMap.put(TargetMethod.TEST_STREAM.name(), new  PtpStreamTestService());
//    }


    protected void configure() {

        Matcher<Class> subpacket = Matchers.inSubpackage("org.fedai");
//        ProvisionListener listener = new ProvisionListener() {
//            @Override
//            public <T> void onProvision(ProvisionInvocation<T> provision) {
//                Key key = provision.getBinding().getKey();
//                System.err.println("guice  key "+key);
//                Class rawType = key.getTypeLiteral().getRawType();
//                if (rawType != null && subpacket.matches(rawType)) {
//                //    System.err.println("xxxxxxxxxxxx"+rawType);
//                //    rawType
//
//                    //   key.getTypeLiteral().getRawType().
////                    Method[] methods = rawType.getMethods();
////                    Arrays.stream(methods).forEach(method -> {
////                        try {
////                            Schedule config = method.getDeclaredAnnotation(Schedule.class);
////                            if (config != null) {
//////                                String methodName = method.getName();
//////                                Class clazz = field.getType();
////                                ScheduleInfo scheduleInfo = new ScheduleInfo();
////                                scheduleInfo.setKey(key);
////                                scheduleInfo.setMethod(method);
////                                scheduleInfo.setCron(config.cron());
////                                Quartz.sheduleInfoMap.put(rawType.getName() + "_" + method.getName(), scheduleInfo);
////
////                            }
////                        } catch (Exception e) {
////                            e.printStackTrace();
////                        }
////                    });
//                    System.err.println(key.getTypeLiteral().getRawType().getName());
//                }
//            }
//        };

//        TypeListener  typeListener = new TypeListener() {
//            @Override
//            public <I> void hear(TypeLiteral<I> type, TypeEncounter<I> encounter) {
//                    System.err.println("====type====="+type);
//            }
//        };
//
//     this.bindListener(Matchers.any(),typeListener);
//     this.bindListener(Matchers.any(),listener);

    }

}
