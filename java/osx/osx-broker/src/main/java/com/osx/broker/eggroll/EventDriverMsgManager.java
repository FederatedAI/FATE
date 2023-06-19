package com.osx.broker.eggroll;

import com.google.common.collect.Lists;
import com.lmax.disruptor.BlockingWaitStrategy;
import com.lmax.disruptor.dsl.Disruptor;
import com.lmax.disruptor.dsl.ProducerType;
import com.lmax.disruptor.util.DaemonThreadFactory;
import com.osx.broker.ServiceContainer;
import com.osx.broker.callback.CreateUserCallback;
//import com.osx.broker.callback.MockDesGrpcEventHandler;
import com.osx.broker.callback.MsgEventCallback;
import com.osx.broker.callback.MsgEventDispatchCallback;
import com.osx.broker.consumer.ConsumerManager;
import com.osx.broker.message.Message;
import com.osx.broker.queue.TransferQueue;
import com.osx.broker.queue.TransferQueueManager;
import com.osx.core.constant.Dict;
import com.osx.core.frame.Lifecycle;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.ConcurrentHashMap;

public class EventDriverMsgManager implements Lifecycle {

    Logger logger  = LoggerFactory.getLogger(EventDriverMsgManager.class);
    ConsumerManager consumerManager=null;
    TransferQueueManager transferQueueManager=null;
    public EventDriverMsgManager(ConsumerManager consumerManager,TransferQueueManager transferQueueManager){
        this.consumerManager = consumerManager;
        this.transferQueueManager = transferQueueManager;
    }




    @Override
    public void init() {
        MsgEventDispatchCallback dispatchCallback = new MsgEventDispatchCallback();
        ServiceContainer.transferQueueManager.addMsgCallBackRule((queue -> {
            if(queue.getTransferId().startsWith(Dict.STREAM_SEND_TOPIC_PREFIX)){
                return true;
            }
            return false;
        }), Lists.newArrayList(new CreateUserCallback(PushEventHandler.class),dispatchCallback));
        ServiceContainer.transferQueueManager.addMsgCallBackRule((queue -> {
            if(queue.getTransferId().startsWith(Dict.STREAM_BACK_TOPIC_PREFIX)){
                return true;
            }
            return false;
        }), Lists.newArrayList(dispatchCallback));

    }





    @Override
    public void start() {


    }

    @Override
    public void destroy() {

    }
}
