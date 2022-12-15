package com.osx.broker.consumer;



import com.google.common.collect.Maps;
import com.osx.core.frame.ServiceThread;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledThreadPoolExecutor;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

public class ConsumerManager  {
    Logger logger = LoggerFactory.getLogger(ConsumerManager.class);
    ScheduledExecutorService scheduledExecutorService = new ScheduledThreadPoolExecutor(1);
//    ConcurrentHashMap<String, Map<Long ,Consumer>>  transferQueueConsumerMap = new ConcurrentHashMap<>();
    ConcurrentHashMap<String, UnaryConsumer> unaryConsumerMap = new ConcurrentHashMap<>();
   // ConcurrentHashMap<String, PushConsumer>  pushConsumerMap  = new ConcurrentHashMap<>();
    ConcurrentHashMap<String, StreamConsumer> streamConsumerMap = new ConcurrentHashMap<>();
    ConcurrentHashMap<String , RedirectConsumer>  redirectConsumerMap = new ConcurrentHashMap<>();
    AtomicLong  consumerIdIndex  = new AtomicLong(0);

//    public  Map<Long,Consumer>  getTransferQueueConsumerSet(String transferId){
//        return transferQueueConsumerMap.get(transferId);
//    }




    public ConsumerManager(){
//        scheduledExecutorService.scheduleAtFixedRate(()->{
//            checkAndClean();
//        },1,1, TimeUnit.SECONDS);

        longPullingThread.start();
       // monitorThread.start();
    }


    public  Map<String,UnaryConsumer> getUnaryConsumerMap(){
           return Maps.newHashMap( this.unaryConsumerMap);
    }


//    ServiceThread  monitorThread  = new ServiceThread() {
//        @Override
//        public String getServiceName() {
//            return "monitorThread";
//        }
//
//        @Override
//        public void run() {
//            AtomicInteger longPullingSize = new AtomicInteger(0);
//            int consumerSize=0;
//            while(true){
//
//                this.waitForRunning(30000);
//            }
//        }
//    };

    public  static  class ReportData{


    }

    public  void   report(){
        AtomicInteger longPullingSize = new AtomicInteger(0);
        longPullingSize.set(0);
        unaryConsumerMap.forEach((transferId, unaryConsumer) -> {
            longPullingSize.addAndGet(unaryConsumer.getLongPullingQueueSize());
        });
        logger.info("consumer monitor,long pulling waiting {} ,total num {}", longPullingSize.get(), unaryConsumerMap.size());

    }


    ServiceThread longPullingThread = new ServiceThread() {
        @Override
        public String getServiceName() {
            return "longPullingThread";
        }
        @Override
        public void run() {
            int interval = 200;
            final AtomicInteger longPullingWaitingSize = new  AtomicInteger(0);
            final AtomicInteger answerCount = new  AtomicInteger(0);
            while(true){
                try {
                    longPullingWaitingSize.set(0);
                    answerCount.set(0);
                    unaryConsumerMap.forEach((transferId, unaryConsumer) -> {
                        try {
                            answerCount.addAndGet(unaryConsumer.answerLongPulling());
                            longPullingWaitingSize.addAndGet(unaryConsumer.getLongPullingQueueSize());
                        }catch(Exception e){
                            e.printStackTrace();
                        }
                    });
                    if (longPullingWaitingSize.get() > 0) {
                        interval = 500;
                    } else {
                        interval = 1000;
                    }
                }catch(Exception igore){
                    igore.printStackTrace();
                }
                this.waitForRunning(interval);
            }

        }
    };

    public  UnaryConsumer  getUnaryConsumer(String  transferId){
        return  unaryConsumerMap.get(transferId);
    }

    public    UnaryConsumer getOrCreateUnaryConsumer(String transferId){
            if (unaryConsumerMap.get(transferId) == null) {
                UnaryConsumer unaryConsumer =
                        new UnaryConsumer(consumerIdIndex.get(), transferId);
                unaryConsumerMap.putIfAbsent(transferId,unaryConsumer);
                return unaryConsumerMap.get(transferId);
            } else {
              return   unaryConsumerMap.get(transferId);
            }
    }



    public  StreamConsumer  getOrCreateStreamConsumer(String transferId){

        if(streamConsumerMap.get(transferId)==null){
            StreamConsumer streamConsumer =   new StreamConsumer(consumerIdIndex.get(),transferId);
            streamConsumerMap.putIfAbsent(transferId,streamConsumer);
            return  streamConsumerMap.get(transferId);
        }else{
            return  streamConsumerMap.get(transferId);
        }
    }


//    public synchronized PushConsumer  getOrCreatePushConsumer(String transferId){
//        if (pushConsumerMap.get(transferId) == null) {
//            PushConsumer pushConsumer =
//                    new PushConsumer(consumerIdIndex.get(), transferId);
//            pushConsumerMap.putIfAbsent(transferId,pushConsumer);
//            return pushConsumerMap.get(transferId);
//        } else {
//            return   pushConsumerMap.get(transferId);
//        }
//    }

    public  synchronized   RedirectConsumer   getOrCreateRedirectConsumer(String  resource){
        logger.info("getOrCreateRedirectConsumer {}",resource);
        if (unaryConsumerMap.get(resource) == null) {
            RedirectConsumer redirectConsumer =
                    new RedirectConsumer(consumerIdIndex.get(), resource);
            unaryConsumerMap.putIfAbsent(resource,redirectConsumer);
            return (RedirectConsumer)unaryConsumerMap.get(resource);
        } else {
            return  (RedirectConsumer) unaryConsumerMap.get(resource);
        }
    }





    public void onComplete(String transferId){
        this.unaryConsumerMap.remove(transferId);
        logger.info("remove consumer {}",transferId);
    }

    /**
     *
     */
    private  void   checkAndClean(){
    }


}
