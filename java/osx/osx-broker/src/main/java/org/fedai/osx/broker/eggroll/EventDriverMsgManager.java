//package org.fedai.osx.broker.eggroll;
//
//import com.google.common.collect.Lists;
//
//import org.fedai.osx.broker.callback.CreateUserCallback;
//
//import org.fedai.osx.broker.consumer.ConsumerManager;
//import org.fedai.osx.broker.queue.TransferQueueManager;
//import org.fedai.osx.core.constant.Dict;
//import org.fedai.osx.core.frame.Lifecycle;
//import org.slf4j.Logger;
//import org.slf4j.LoggerFactory;
//
//public class EventDriverMsgManager implements Lifecycle {
//
//    Logger logger  = LoggerFactory.getLogger(EventDriverMsgManager.class);
//    ConsumerManager consumerManager=null;
//    TransferQueueManager transferQueueManager=null;
//    public EventDriverMsgManager(ConsumerManager consumerManager,TransferQueueManager transferQueueManager){
//        this.consumerManager = consumerManager;
//        this.transferQueueManager = transferQueueManager;
//    }
//
//
//
//
//    @Override
//    public void init() {
//        MsgEventDispatchCallback dispatchCallback = new MsgEventDispatchCallback();
//        transferQueueManager.addMsgCallBackRule((queue -> {
//            if(queue.getTransferId().startsWith(Dict.STREAM_SEND_TOPIC_PREFIX)){
//                return true;
//            }
//            return false;
//        }), Lists.newArrayList(new CreateUserCallback(PushEventHandler.class),dispatchCallback));
//        transferQueueManager.addMsgCallBackRule((queue -> {
//            if(queue.getTransferId().startsWith(Dict.STREAM_BACK_TOPIC_PREFIX)){
//                return true;
//            }
//            return false;
//        }), Lists.newArrayList(dispatchCallback));
//
//    }
//
//
//
//
//
//    @Override
//    public void start() {
//
//
//    }
//
//    @Override
//    public void destroy() {
//
//    }
//}
