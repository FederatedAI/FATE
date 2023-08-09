package org.fedai.osx.broker.callback;

import org.fedai.osx.broker.ServiceContainer;
import org.fedai.osx.broker.consumer.EventDrivenConsumer;
import org.fedai.osx.broker.message.MessageExt;
import org.fedai.osx.broker.queue.TransferQueue;

public class MsgEventDispatchCallback implements MsgEventCallback{



    @Override
    public void callback(TransferQueue transferQueue, MessageExt message) throws Exception {

        String topic = transferQueue.getTransferId();
        EventDrivenConsumer eventDrivenConsumer = ServiceContainer.consumerManager.getEventDrivenConsumer(topic);
        if(eventDrivenConsumer!=null){
            if(!transferQueue.isHasEventMsgDestoryCallback()) {
                transferQueue.registerDestoryCallback(() -> {
                    ServiceContainer.consumerManager.onComplete(topic);
                });
                transferQueue.setHasEventMsgDestoryCallback(true);
            }
//            MessageEvent  messageEvent = new MessageEvent();
//            messageEvent.setTopic(topic);
//
//            messageEvent.setDesComponent(message.getProperty(Dict.DES_COMPONENT));
//            messageEvent.setSrcComponent(message.getProperty(Dict.SOURCE_COMPONENT));
//            messageEvent.setSrcPartyId(message.getSrcPartyId());
//            messageEvent.setDesPartyId(message.getDesPartyId());
//            messageEvent.setSessionId(message.getProperty(Dict.SESSION_ID));
            eventDrivenConsumer.fireEvent(message);
        }
    }
}
