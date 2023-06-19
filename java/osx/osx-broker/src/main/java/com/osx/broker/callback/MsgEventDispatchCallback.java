package com.osx.broker.callback;

import com.osx.broker.ServiceContainer;
import com.osx.broker.consumer.EventDrivenConsumer;
import com.osx.broker.consumer.MessageEvent;
import com.osx.broker.message.Message;
import com.osx.broker.message.MessageExt;
import com.osx.broker.queue.TransferQueue;
import com.osx.core.constant.Dict;

import javax.xml.ws.Service;

public class MsgEventDispatchCallback implements MsgEventCallback{
    @Override
    public void callback(TransferQueue transferQueue, MessageExt message) {
        String topic = transferQueue.getTransferId();
        EventDrivenConsumer  eventDrivenConsumer = ServiceContainer.consumerManager.getEventDrivenConsumer(topic);
        if(eventDrivenConsumer!=null){
            MessageEvent  messageEvent = new MessageEvent();
            messageEvent.setTopic(topic);
            messageEvent.setDesComponent(message.getProperty(Dict.DES_COMPONENT));
            messageEvent.setSrcComponent(message.getProperty(Dict.SOURCE_COMPONENT));
            messageEvent.setSrcPartyId(message.getSrcPartyId());
            messageEvent.setDesPartyId(message.getDesPartyId());
            messageEvent.setSessionId(message.getProperty(Dict.SESSION_ID));
            eventDrivenConsumer.fireEvent(messageEvent);
        }
    }
}
