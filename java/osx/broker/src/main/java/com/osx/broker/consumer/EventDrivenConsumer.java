package com.osx.broker.consumer;

import com.lmax.disruptor.BlockingWaitStrategy;
import com.lmax.disruptor.EventHandler;
import com.lmax.disruptor.EventTranslatorOneArg;
import com.lmax.disruptor.dsl.Disruptor;
import com.lmax.disruptor.dsl.ProducerType;
import com.lmax.disruptor.util.DaemonThreadFactory;


public  class EventDrivenConsumer extends LocalQueueConsumer  {

    EventHandler  eventHandler;
    Disruptor  disruptor;



    public EventDrivenConsumer(long consumerId, String topic,EventHandler eventHandler){

        super(consumerId,topic);
        this.eventHandler = eventHandler;
        disruptor = new Disruptor(() -> new MessageEvent(),
                2048, DaemonThreadFactory.INSTANCE,
                ProducerType.SINGLE, new BlockingWaitStrategy());
        disruptor.handleEventsWith(eventHandler);
        disruptor.start();

    }
    public static final EventTranslatorOneArg<MessageEvent,MessageEvent> TRANSLATOR =
            (event, sequence, arg) -> {
                event.setTopic(arg.getTopic());
                //event.setIndex(arg.getIndex());
            };

    public  void  fireEvent(MessageEvent event){
        disruptor.publishEvent((EventTranslatorOneArg) TRANSLATOR,event);
    }




}
