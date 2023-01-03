package com.osx.broker;


import com.osx.broker.consumer.ConsumerManager;
import com.osx.broker.grpc.ClusterService;
import com.osx.broker.grpc.PcpGrpcService;
import com.osx.broker.grpc.ProxyGrpcService;
import com.osx.broker.interceptor.RequestHandleInterceptor;
import com.osx.broker.message.AllocateMappedFileService;
import com.osx.broker.queue.TransferQueueManager;
import com.osx.broker.router.DefaultFateRouterServiceImpl;
import com.osx.broker.router.FateRouterService;
import com.osx.broker.server.OsxServer;
import com.osx.broker.service.PushService2;
import com.osx.broker.service.TokenApplyService;
import com.osx.broker.service.UnaryCallService;
import com.osx.broker.store.MessageStore;
import com.osx.broker.token.DefaultTokenService;
import com.osx.broker.zk.CuratorZookeeperClient;
import com.osx.broker.zk.ZkConfig;
import com.osx.core.config.MetaInfo;
import com.osx.core.flow.ClusterFlowRuleManager;
import com.osx.core.flow.FlowCounterManager;
import com.osx.core.service.AbstractServiceAdaptor;
import com.osx.tech.provider.TechProviderRegister;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.HashMap;
import java.util.Map;

public class ServiceContainer {
    static public ConsumerManager consumerManager;
    static public PcpGrpcService pcpGrpcService;
    static public TransferQueueManager transferQueueManager;
    static public AllocateMappedFileService allocateMappedFileService;
    static public FlowCounterManager flowCounterManager;
    // static public ZookeeperRegistry  zookeeperRegistry;
    static public OsxServer transferServer;
    static public ProxyGrpcService proxyGrpcService;
    static public FateRouterService fateRouterService;
    //    static public QueueGrpcService queueGrpcservice;
    // static public CommonService commonService;
    static public ClusterService clusterService;
    //static public ProducerStreamService  producerStreamService;
    static public Map<String, AbstractServiceAdaptor> serviceAdaptorMap = new HashMap<String, AbstractServiceAdaptor>();
    //static public DLedgerServer dLedgerServer;
    static public TokenApplyService tokenApplyService;
    static public PushService2 pushService2;
    static public UnaryCallService unaryCallService;
    static public RequestHandleInterceptor requestHandleInterceptor;
    // static public ConsumeUnaryService  consumeUnaryService;
//    static public CancelTransferService  cancelTransferService;
//    static public AckService   ackService;
//    static public QueryTransferQueueService queryTransferQueueService;
    static public MessageStore messageStore;
//    static public DefaultRouterInterceptor defaultRouterInterceptor;
    //   static public ProducerUnaryService producerUnaryService;
    // static public SyncQueueService syncQueueService;
    //static public ClusterClientEndpoint clusterClientEndpoint;
    //static public ReportService  reportService;
//    static public RedirectSinker redirectSinker;
    static public ClusterFlowRuleManager clusterFlowRuleManager;
    static public DefaultTokenService defaultTokenService;
    //static public TokenApplyService  tokenApplyService;
    static public CuratorZookeeperClient zkClient;
    static public TechProviderRegister techProviderRegister;
//    static public ClusterQueueApplyService  clusterQueueApplyService;
    static Logger logger = LoggerFactory.getLogger(ServiceContainer.class);

    public static void init() {
        flowCounterManager = createFlowCounterManager();
        clusterFlowRuleManager = createClusterFlowRuleManager();
        //zookeeperRegistry = createServiceRegistry();
        allocateMappedFileService = createAllocateMappedFileService();
        messageStore = createMessageStore(allocateMappedFileService);
        zkClient = createCuratorZookeeperClient();
        transferQueueManager = createTransferQueueManager();
        consumerManager = createTransferQueueConsumerManager();
        fateRouterService = createFateRouterService();
        tokenApplyService = createTokenApplyService();
        pushService2 = createPushService2();
        // consumeUnaryService = createConsumeUnaryService();
        // cancelTransferService = createCancelTransferService(transferQueueManager,consumerManager);
        //producerStreamService = createProducerStreamService(tokenApplyService,fateRouterService,consumerManager,transferQueueManager);
        //   producerUnaryService = createProducerUnaryService(fateRouterService,consumerManager,transferQueueManager);
        //queryTransferQueueService = new QueryTransferQueueService(transferQueueManager);
        //  queueGrpcservice =       createQueueGrpcservice();
        requestHandleInterceptor = createDefaulRequestInterceptor(fateRouterService);
//        defaultRouterInterceptor = createDefaultRouterInterceptor(fateRouterService);
        unaryCallService = createUnaryCallService(requestHandleInterceptor);
        proxyGrpcService = new ProxyGrpcService(pushService2, unaryCallService);
        transferServer = new OsxServer();
        //clusterClientEndpoint = createClusterClientEndpoint();
        //reportService = createReportService();
        //ackService = createAckService();
        // commonService = createCommonService();
//        redirectSinker = createRedirectSinker();
        defaultTokenService = createDefaultTokenService();
        tokenApplyService = createTokenApplyService();
        //syncQueueService = createSyncQueueService();
        // clusterQueueApplyService = createClusterQueueApplyService();
        clusterService = createClusterService();
        // dLedgerServer = createDLedgerServer();

        pcpGrpcService = createPcpGrpcService();
        techProviderRegister = createTechProviderRegister();
        if (!transferServer.start()) {
            System.exit(-1);
        } else {

        }
        ;


    }

    public static TechProviderRegister createTechProviderRegister() {
        TechProviderRegister techProviderRegister = new TechProviderRegister();
        techProviderRegister.init();
        return techProviderRegister;
    }

    public static PcpGrpcService createPcpGrpcService() {

        return new PcpGrpcService();
    }

    public static ClusterService createClusterService() {
        return new ClusterService();
    }

//    public  static ClusterQueueApplyService createClusterQueueApplyService(){
//        return  new  ClusterQueueApplyService();
//    };


    //public  static  SyncQueueService  createSyncQueueService(){
//            return  new  SyncQueueService();
//    }

    public static CuratorZookeeperClient createCuratorZookeeperClient() {
        if (MetaInfo.isCluster()) {
            ZkConfig zkConfig = new ZkConfig(MetaInfo.PROPERTY_ZK_URL, 5000);
            return new CuratorZookeeperClient(zkConfig);
        }
        return null;
    }

    public static TokenApplyService createTokenApplyService() {
        TokenApplyService tokenApplyService = new TokenApplyService();
        tokenApplyService.start();
        return tokenApplyService;
    }

    public static DefaultTokenService createDefaultTokenService() {
        return new DefaultTokenService();
    }
//   // public  static CommonService createCommonService(){
//        return  new CommonService();
//    };

//    public  static RedirectSinker createRedirectSinker(){
//        return  new  RedirectSinker();
//    }

    public static ClusterFlowRuleManager createClusterFlowRuleManager() {
        return new ClusterFlowRuleManager();
    }

//    public static AckService  createAckService(){
//        AckService   ackService = new  AckService();
//        return   ackService;
//    }

//    public static ClusterClientEndpoint  createClusterClientEndpoint(){
//        ClusterClientEndpoint  clusterClientEndpoint = new ClusterClientEndpoint();
//        clusterClientEndpoint.start();
//        return clusterClientEndpoint;
//    }


//    public  static  ReportService createReportService(){
//        ReportService  reportService = new ReportService();
//         return reportService;
//    }


    public static MessageStore createMessageStore(
            AllocateMappedFileService allocateMappedFileService) {
        // TransferQueueManager transferQueueManager ,AllocateMappedFileService allocateMappedFileService,String path){
        MessageStore messageStore = new MessageStore(allocateMappedFileService
                , MetaInfo.PROPERTY_TRANSFER_FILE_PATH_PRE + File.separator + MetaInfo.INSTANCE_ID + File.separator + "message-store");
        messageStore.start();
        return messageStore;

    }


//    public static QueueGrpcService  createQueueGrpcservice(
//                                                           ){
//      return    new QueueGrpcService();
//    }

    public static RequestHandleInterceptor createDefaulRequestInterceptor(FateRouterService routerService) {
        RequestHandleInterceptor requestHandleInterceptor = new RequestHandleInterceptor(routerService);
        return requestHandleInterceptor;
    }


    static FlowCounterManager createFlowCounterManager() {
        FlowCounterManager flowCounterManager = new FlowCounterManager("transfer");
        flowCounterManager.startReport();
        return flowCounterManager;
    }

    static UnaryCallService createUnaryCallService(RequestHandleInterceptor requestHandleInterceptor) {
        UnaryCallService unaryCallService = new UnaryCallService();
        unaryCallService.addPreProcessor(requestHandleInterceptor);
        return unaryCallService;
    }

    static PushService2 createPushService2() {
        PushService2 pushService2 = new PushService2();
        return pushService2;
    }


//    static  ZookeeperRegistry createServiceRegistry() {
//        Preconditions.checkArgument(StringUtils.isNotEmpty(MetaInfo.PROPERTY_ZK_URL));
//        return ZookeeperRegistry.createRegistry(MetaInfo.PROPERTY_ZK_URL, Dict.SERVICE_FIREWORK, Dict.ONLINE_ENVIRONMENT, MetaInfo.PROPERTY_PORT);
//    }


//    static RouterService createRouterService(ZookeeperRegistry zookeeperRegistry) {
//        DefaultRouterService routerService = new DefaultRouterService();
//        routerService.setRegistry(zookeeperRegistry);
//        return routerService;
//    }


//    static ProducerUnaryService createProducerUnaryService(
//                                                       FateRouterService fateRouterService,
//                                                       ConsumerManager consumerManager,
//                                                       TransferQueueManager transferQueueManager
//    ){
//        ProducerUnaryService  producerService = new ProducerUnaryService(
//                fateRouterService,
//                consumerManager,
//                transferQueueManager);
//        return  producerService;
//    }

//    static ConsumeUnaryService createConsumeUnaryService(
//    ){
//        ConsumeUnaryService  consumeUnaryService = new ConsumeUnaryService();
//        return  consumeUnaryService;
//    }


//    static CancelTransferService createCancelTransferService(
//                                                         TransferQueueManager transferQueueManager,
//                                                         ConsumerManager  consumerManager
//    ){
//        CancelTransferService  service = new CancelTransferService(transferQueueManager,consumerManager);
//        return  service;
//    }


//    static ProducerStreamService createProducerStreamService(TokenApplyService tokenApplyService,
//                                                       FateRouterService fateRouterService,
//                                                       ConsumerManager consumerManager,
//                                                       TransferQueueManager transferQueueManager
//    ){
//        ProducerStreamService  producerService = new ProducerStreamService( tokenApplyService,
//                fateRouterService,
//                consumerManager,
//                transferQueueManager);
//        return  producerService;
//    }

    static ConsumerManager createTransferQueueConsumerManager() {
        ConsumerManager consumerManager = new ConsumerManager();
        return consumerManager;
    }


//    static  TokenApplyService createTokenApplyService(){
//        TokenApplyService   tokenApplyService = new  TokenApplyService();
//        tokenApplyService.start();
//        return tokenApplyService;
//    }

    static FateRouterService createFateRouterService() {
        DefaultFateRouterServiceImpl fateRouterService = new DefaultFateRouterServiceImpl();
        fateRouterService.start();
        return fateRouterService;
    }

    static TransferQueueManager createTransferQueueManager() {
        TransferQueueManager transferQueueManager = new TransferQueueManager();
        return transferQueueManager;
    }

    static AllocateMappedFileService createAllocateMappedFileService() {
        AllocateMappedFileService allocateMappedFileService = new AllocateMappedFileService();
        allocateMappedFileService.start();
        return allocateMappedFileService;
    }


//    public static void  handleServiceAdaptor() {
//
//        Reflections reflections = new Reflections("com.firework");
//
//        Set<Class<? extends Interceptor>> interceptors = reflections.getSubTypesOf(Interceptor.class);
//
//        Set<Class<?>> sets = reflections.getTypesAnnotatedWith(FateService.class);
//
//
//        for (Class<?> clazz : sets) {
//            try {
//
//                AbstractServiceAdaptor abstractServiceAdaptor = (AbstractServiceAdaptor) clazz.newInstance();
//                FateService fateService = clazz.getAnnotation(FateService.class);
//                String name = fateService.name();
//                serviceAdaptorMap.put(name, abstractServiceAdaptor);
//                Class[] preChainClasses = fateService.preChain();
//
//
//                if (preChainClasses != null) {
//                    for (Class interceptor : preChainClasses) {
//                        abstractServiceAdaptor.addPreProcessor((Interceptor) interceptor.newInstance());
//                        ;
//                    }
//                }
//
//                Class[] postChainClasses = fateService.postChain();
//
//                if (postChainClasses != null) {
//                    for (Class interceptor : postChainClasses) {
//                        abstractServiceAdaptor.addPostProcessor((Interceptor) interceptor.newInstance());
//                        ;
//                    }
//                }
//
//            } catch (InstantiationException e) {
//                e.printStackTrace();
//            } catch (IllegalAccessException e) {
//                e.printStackTrace();
//            }
//        }
//
//        logger.info("===== {}",serviceAdaptorMap);
//    }


//    public ApplicationListener<ApplicationReadyEvent> registerComponent(ZookeeperRegistry zookeeperRegistry) {
//        return applicationReadyEvent -> zookeeperRegistry.registerComponent();
//    }
//

}
