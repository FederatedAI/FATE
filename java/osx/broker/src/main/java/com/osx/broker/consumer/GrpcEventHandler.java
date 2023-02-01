package com.osx.broker.consumer;

import com.lmax.disruptor.EventHandler;
import com.osx.broker.ServiceContainer;
import com.osx.broker.message.MessageExt;
import com.osx.broker.queue.TransferQueue;
import com.osx.core.constant.StatusCode;
import com.osx.core.context.Context;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public abstract class GrpcEventHandler implements EventHandler<MessageEvent> {

    Logger logger = LoggerFactory.getLogger(GrpcEventHandler.class);

    @Override
    public void onEvent(MessageEvent event, long l, boolean b) throws Exception {
        String  topic =  event.getTopic();
        EventDrivenConsumer consumer = ServiceContainer.consumerManager.getEventDrivenConsumer(topic);
        TransferQueue.TransferQueueConsumeResult transferQueueConsumeResult = consumer.consume(new Context(), -1);
        if (transferQueueConsumeResult.getCode().equals(StatusCode.SUCCESS)) {
            long index = transferQueueConsumeResult.getRequestIndex();
            //ack 的位置需要调整
            consumer.ack(index);
            MessageExt messageExt = transferQueueConsumeResult.getMessage();
            int flag = messageExt.getFlag();
            switch (flag){
                //msg
                case 0: handleMessage(messageExt);break;
                //error
                case 1: handleError(messageExt);break;
                //completed
                case 2: handleComplete(messageExt);break;
                default:;
            }
        }else{
            logger.warn("");
        }
    }

    protected abstract void  handleMessage(MessageExt message);
    protected abstract void  handleError(MessageExt message);
    protected abstract void  handleComplete(MessageExt message);
}
