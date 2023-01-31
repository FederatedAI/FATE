package com.osx.broker.callback;

import com.osx.broker.ServiceContainer;
import com.osx.broker.consumer.EventDrivenConsumer;
import com.osx.broker.consumer.MessageEvent;
import com.osx.broker.message.Message;
import com.osx.broker.queue.TransferQueue;

import javax.xml.ws.Service;

public class MsgEventDispatchCallback implements MsgEventCallback{
    @Override
    public void callback(TransferQueue transferQueue, Message message) {
        String topic = transferQueue.getTransferId();
        EventDrivenConsumer  eventDrivenConsumer = ServiceContainer.consumerManager.getEventDrivenConsumer(topic);
        if(eventDrivenConsumer!=null){
            MessageEvent  messageEvent = new MessageEvent();
            eventDrivenConsumer.fireEvent(messageEvent);
        }
    }
}
