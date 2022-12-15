//package com.firework.transfer.grpc;
//import com.firework.core.bean.RouterInfo;
//import com.firework.core.constant.TransferStatus;
//import com.firework.core.context.Context;
//import com.firework.transfer.buffer.TransferBuffer;
//import com.firework.transfer.buffer.TransferBufferPool;
//import com.firework.transfer.buffer.WriteResult;
//import com.firework.transfer.buffer.WriteStatus;
//import com.firework.transfer.constants.Direction;
//import com.firework.transfer.event.PushEventHandler;
//import com.firework.transfer.event.TransferEvent;
//import com.firework.transfer.exception.NoRouterInfoException;
//import com.firework.transfer.router.FateRouterService;
//import com.firework.transfer.service.TokenApplyService;
//import com.firework.transfer.util.ResourceUtil;
//import com.lmax.disruptor.BlockingWaitStrategy;
//import com.lmax.disruptor.EventFactory;
//import com.lmax.disruptor.RingBuffer;
//import com.lmax.disruptor.dsl.Disruptor;
//import com.lmax.disruptor.dsl.ProducerType;
//import com.webank.ai.eggroll.api.networking.proxy.DataTransferServiceGrpc;
//import com.webank.ai.eggroll.api.networking.proxy.Proxy;
//import io.grpc.ManagedChannel;
//import io.grpc.stub.StreamObserver;
//import org.slf4j.Logger;
//import org.slf4j.LoggerFactory;
//
//import java.util.concurrent.ArrayBlockingQueue;
//import java.util.concurrent.BlockingQueue;
//import java.util.concurrent.CountDownLatch;
//import java.util.concurrent.ThreadFactory;
//
//public class LocalCachePushReqStreamObserver implements StreamObserver<Proxy.Packet> {
//
//    Logger logger = LoggerFactory.getLogger(LocalCachePushReqStreamObserver.class);
//
//    TransferStatus transferStatus = TransferStatus.INIT;
//
//    Context context;
//
//    FateRouterService  fateRouterService;
//
//    private   TransferBufferPool  transferBufferPool;
//
//
//    BlockingQueue<TransferBuffer> borrowed = new ArrayBlockingQueue<TransferBuffer>(100);
//
//    CountDownLatch  finishLatch ;
//
//    TokenApplyService  tokenApplyService ;
//
//    private StreamObserver<Proxy.Packet> forwardPushReqSO;
//
//    private  StreamObserver<Proxy.Metadata>  backRespSO;
//
//    private String  transferId;
//
//    public LocalCachePushReqStreamObserver(Context  context, TransferBufferPool transferBufferPool,
//                                           FateRouterService  fateRouterService,StreamObserver  backRespSO,
//                                           TokenApplyService  tokenApplyService,
//                                           CountDownLatch finishLatch){
//        this.transferId = context.getCaseId();
//        this.fateRouterService = fateRouterService;
//        this.backRespSO =  backRespSO;
//        this.context = context.subContext();
//        this.transferBufferPool = transferBufferPool;
//        this.tokenApplyService = tokenApplyService;
//        this.context.setServiceName("pushTransfer");
//        // 生产者的线程工厂
//        ThreadFactory threadFactory = new ThreadFactory(){
//            @Override
//            public Thread newThread(Runnable r) {
//                return new Thread(r, "simpleThread");
//            }
//        };
//        // RingBuffer生产工厂,初始化RingBuffer的时候使用
//        EventFactory<TransferEvent> factory = new EventFactory<TransferEvent>() {
//            @Override
//            public TransferEvent newInstance() {
//                return new TransferEvent();
//            }
//        };
//        // 阻塞策略
//        BlockingWaitStrategy strategy = new BlockingWaitStrategy();
//        // 创建disruptor，采用单生产者模式
//        disruptor = new Disruptor(factory, 64, threadFactory, ProducerType.SINGLE, strategy);
//        this.finishLatch = finishLatch;
//    }
//
//    public StreamObserver<Proxy.Packet> getForwardPushReqSO() {
//        return forwardPushReqSO;
//    }
//
//    public void setForwardPushReqSO(StreamObserver<Proxy.Packet> forwardPushReqSO) {
//        this.forwardPushReqSO = forwardPushReqSO;
//    }
//
//    private TransferBuffer borrowTransferBuffer(int size,String transferId){
//        TransferBuffer  transferBuffer =transferBufferPool.borrowBuffer(size,transferId);
//        if(transferBuffer!=null){
//            borrowed.add(transferBuffer);
//        }else{
//            throw  new RuntimeException();
//        }
//        return transferBuffer;
//    }
//
//
//    public   void  init(Proxy.Packet  packet,int size )  {
//
//        RouterInfo routerInfo = fateRouterService.route(packet);
//        if(routerInfo!=null) {
//            context.setRouterInfo(routerInfo);
//            context.setSrcPartyId(routerInfo.getSourcePartyId());
//            context.setDesPartyId(routerInfo.getDesPartyId());
//        }else{
//            throw  new NoRouterInfoException("no router");
//        }
//        /**
//         * 以下代码顺序有严格要求
//         */
//        transferBuffer = borrowTransferBuffer(size,transferId);
//        ManagedChannel managedChannel = GrpcConnectionPool.getPool().getManagedChannel(context.getRouterInfo().getHost(),context.getRouterInfo().getPort());
//        //logger.info("channel status {}",managedChannel(true).toString());
//        DataTransferServiceGrpc.DataTransferServiceStub stub = DataTransferServiceGrpc.newStub(managedChannel);
//
//        ForwardPushRespSO forwardPushRespSO = new  ForwardPushRespSO(context,backRespSO,()->{finishLatch.countDown();clear();context.printFlowLog();},
//                (e)->{finishLatch.countDown();clear();context.printFlowLog();});
//        forwardPushRespSO.setTokenApplyService(tokenApplyService);
//        forwardPushReqSO =  stub.push(forwardPushRespSO);
//        disruptor.handleEventsWith(new PushEventHandler(transferId,forwardPushReqSO,backRespSO,tokenApplyService,()->{
//            clear();
//        },(t)->{
//            logger.error("error callback",t);
//            transferStatus=TransferStatus.ERROR;
//            clear();
//        }));
//        disruptor.start();
//
////        logger.info("borrowed {}",borrowed);
//        transferStatus = TransferStatus.OK;
//    }
//
//    TransferBuffer transferBuffer;
//
//    Disruptor<TransferEvent> disruptor;
//
//    @Override
//    public void onNext(Proxy.Packet value) {
//        try {
//            if (transferStatus.equals(TransferStatus.ERROR)) {
//                throw new RuntimeException("transfer error");
//            }
//            byte[] data = value.toByteArray();
//            int size = data.length;
//            if (transferStatus.equals(TransferStatus.INIT)) {
//                init(value, size);
//            }
//            RouterInfo routerInfo = context.getRouterInfo();
//            String sourcePartyId = routerInfo.getSourcePartyId();
//            String resource = ResourceUtil.buildResource(sourcePartyId, Direction.DOWN);
//            this.tokenApplyService.applyToken(context, resource, size);
//            WriteResult writeResult = transferBuffer.write(data, transferId);
//            if (logger.isTraceEnabled()) {
//                logger.info("write cache result {}", writeResult);
//            }
//            boolean writeSuccess = false;
//            if (writeResult.getStatus().equals(WriteStatus.OK)) {
//                writeSuccess = true;
//            } else {
//                logger.error("write cache result {}", writeResult);
//                if (writeResult.getStatus().equals(WriteStatus.FULL)) {
//                    //申请新的缓存
//                    logger.info("cache {} full ", transferBuffer.getBufferId());
//                    transferBuffer = borrowTransferBuffer(size, transferId);
//                    logger.info("borrow new cache {} ", transferBuffer.getBufferId());
//                    writeResult = transferBuffer.write(data, transferId);
//                    ;
//                }
//            }
//            if (writeSuccess) {
//                RingBuffer<TransferEvent> ringBuffer = disruptor.getRingBuffer();
//                long sequence = ringBuffer.next();
//                TransferEvent transferEvent = ringBuffer.get(sequence);
//                transferEvent.setType(TransferEvent.TransferEventType.PUSH_TRANSFER);
//                transferEvent.setBeginIndex(writeResult.getWriteIndex());
//                transferEvent.setSize(size);
//                transferEvent.setTransferBuffer(transferBuffer);
//                transferEvent.setContext(context);
//                ringBuffer.publish(sequence);
//            } else {
//                logger.info("write result {}", writeResult);
//                throw new RuntimeException("write cache result "+writeResult);
//            }
//        }catch (Throwable e){
//            clear();
//            throw e;
//        }
//
//    }
//
//    public void finalize(){
//        if(this.borrowed.size()!=0){
//            try {
//                returnBuffer();
//            }catch(Throwable t){
//
//            }
//        }
//    }
//    public   void  clear(){
//
//        new Thread(()->{
//            disruptor.shutdown();
//            returnBuffer();
//        }).start();
//
//
//    }
//    private  void  returnBuffer(){
//        //logger.info("prepare to return {}",this.borrowed);
//        this.borrowed.forEach(buffer ->{
//            try{
//                buffer.returnBuffer(transferId);
//            }catch(Throwable e){
//                logger.error("return buffer error");
//            }
//        });
//    }
//
//
//
//
//    @Override
//    public void onError(Throwable t) {
//        /**
//         * 传递错误
//         */
//        logger.info("onError",t);
//        context.setException(t);
//        RingBuffer<TransferEvent> ringBuffer = disruptor.getRingBuffer();
//        long sequence = ringBuffer.next();
//        TransferEvent transferEvent = ringBuffer.get(sequence);
//        transferEvent.setError(t);
//        transferEvent.setType(TransferEvent.TransferEventType.PUSH_ERROR);
//        ringBuffer.publish(sequence);
//    }
//
//    @Override
//    public void onCompleted() {
//       // logger.info("receive completed");
//        RingBuffer<TransferEvent> ringBuffer = disruptor.getRingBuffer();
//        long sequence = ringBuffer.next();
//        TransferEvent transferEvent = ringBuffer.get(sequence);
//        transferEvent.setType(TransferEvent.TransferEventType.PUSH_COMPELETED);
//        ringBuffer.publish(sequence);
//    }
//
//
//
////    public  static void  main(String[] args){
////
////
////
////        ThreadFactory threadFactory = new ThreadFactory(){
////            @Override
////            public Thread newThread(Runnable r) {
////                return new Thread(r, "simpleThread");
////            }
////        };
////        // RingBuffer生产工厂,初始化RingBuffer的时候使用
////        EventFactory<TransferEvent> factory = new EventFactory<TransferEvent>() {
////            @Override
////            public TransferEvent newInstance() {
////                return new TransferEvent();
////            }
////        };
////        // 阻塞策略
////        BlockingWaitStrategy strategy = new BlockingWaitStrategy();
////        // 创建disruptor，采用单生产者模式
////        Disruptor disruptor = new Disruptor(factory, 64, threadFactory, ProducerType.SINGLE, strategy);
////        disruptor.handleEventsWith(new EventHandler<TransferEvent>() {
////            @Override
////            public void onEvent(TransferEvent event, long sequence, boolean endOfBatch) throws Exception {
////                if(event.getBeginIndex()%2==0)
////                    Thread.sleep(3000);
////                System.err.print(event.getBeginIndex());   System.err.print(" ");
////            }
////        });
////        disruptor.start();
////
////
////
////
////        new  Thread(new Runnable() {
////            @Override
////            public void run() {
////                for(int i =0;i<100;i++) {
////                    RingBuffer<TransferEvent> ringBuffer = disruptor.getRingBuffer();
////                    long sequence = ringBuffer.next();
////                    TransferEvent transferEvent = ringBuffer.get(sequence);
////                    transferEvent.setType(TransferEvent.TransferEventType.PUSH_TRANSFER);
////                    transferEvent.setBeginIndex(i);
////                    transferEvent.setSize(0);
////                    transferEvent.setTransferBuffer(null);
////                    transferEvent.setContext(null);
////                    ringBuffer.publish(sequence);
////                    try {
////                        Thread.sleep(500);
////                    } catch (InterruptedException e) {
////                        e.printStackTrace();
////                    }
////                }
////            }
////        }).start();
////
////    }
//}
