/*
 * Copyright 2019 The FATE Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.osx.broker.consumer;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.lmax.disruptor.EventHandler;
import com.osx.broker.ServiceContainer;
import com.osx.broker.callback.MsgEventCallback;

import com.osx.broker.message.Message;
import com.osx.broker.queue.TransferQueue;
import com.osx.core.constant.Dict;
import com.osx.core.frame.Lifecycle;
import com.osx.core.frame.ServiceThread;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.awt.*;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledThreadPoolExecutor;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

public class ConsumerManager   implements Lifecycle {
    Logger logger = LoggerFactory.getLogger(ConsumerManager.class);
    ScheduledExecutorService scheduledExecutorService = new ScheduledThreadPoolExecutor(1);
    ConcurrentHashMap<String, UnaryConsumer> unaryConsumerMap = new ConcurrentHashMap<>();
    ConcurrentHashMap<String, EventDrivenConsumer> eventDrivenConsumerMap = new ConcurrentHashMap<>();
    ConcurrentHashMap<String, RedirectConsumer> redirectConsumerMap = new ConcurrentHashMap<>();
    AtomicLong consumerIdIndex = new AtomicLong(0);

    ServiceThread longPullingThread = new ServiceThread() {
        @Override
        public String getServiceName() {
            return "longPullingThread";
        }

        @Override
        public void run() {
            int interval = 200;
            final AtomicInteger longPullingWaitingSize = new AtomicInteger(0);
            final AtomicInteger answerCount = new AtomicInteger(0);
            while (true) {
                try {
                    longPullingWaitingSize.set(0);
                    answerCount.set(0);
                    unaryConsumerMap.forEach((transferId, unaryConsumer) -> {
                        try {
                            answerCount.addAndGet(unaryConsumer.answerLongPulling());
                            longPullingWaitingSize.addAndGet(unaryConsumer.getLongPullingQueueSize());
                        } catch (Exception igore) {

                        }
                    });
                    if (longPullingWaitingSize.get() > 0) {
                        interval = 500;
                    } else {
                        interval = 1000;
                    }
                } catch (Exception igore) {

                }
                this.waitForRunning(interval);
            }

        }
    };


    public ConsumerManager() {
        longPullingThread.start();
    }

    public Map<String, UnaryConsumer> getUnaryConsumerMap() {
        return Maps.newHashMap(this.unaryConsumerMap);
    }

    public void report() {
        AtomicInteger longPullingSize = new AtomicInteger(0);
        longPullingSize.set(0);
        unaryConsumerMap.forEach((transferId, unaryConsumer) -> {
            longPullingSize.addAndGet(unaryConsumer.getLongPullingQueueSize());
        });
        logger.info("consumer monitor,long pulling waiting {} ,total num {}", longPullingSize.get(), unaryConsumerMap.size());

    }

    public UnaryConsumer getUnaryConsumer(String transferId) {
        return unaryConsumerMap.get(transferId);
    }


    public EventDrivenConsumer  getEventDrivenConsumer(String topic){

        return this.eventDrivenConsumerMap.get(topic);

    }

    public EventDrivenConsumer  createEventDrivenConsumer(String topic, EventHandler  eventHandler){
        if (eventDrivenConsumerMap.get(topic) == null) {
            EventDrivenConsumer  eventDrivenConsumer =
                    new EventDrivenConsumer(consumerIdIndex.get(), topic,eventHandler);
            eventDrivenConsumerMap.putIfAbsent(topic, eventDrivenConsumer);
            return eventDrivenConsumerMap.get(topic);
        } else {
            return eventDrivenConsumerMap.get(topic);
        }
    }


    public UnaryConsumer getOrCreateUnaryConsumer(String transferId) {
        if (unaryConsumerMap.get(transferId) == null) {
            UnaryConsumer unaryConsumer =
                    new UnaryConsumer(consumerIdIndex.get(), transferId);
            unaryConsumerMap.putIfAbsent(transferId, unaryConsumer);
            return unaryConsumerMap.get(transferId);
        } else {
            return unaryConsumerMap.get(transferId);
        }
    }

//    public StreamConsumer getOrCreateStreamConsumer(String transferId) {
//
//        if (streamConsumerMap.get(transferId) == null) {
//            StreamConsumer streamConsumer = new StreamConsumer(consumerIdIndex.get(), transferId);
//            streamConsumerMap.putIfAbsent(transferId, streamConsumer);
//            return streamConsumerMap.get(transferId);
//        } else {
//            return streamConsumerMap.get(transferId);
//        }
//    }

    public synchronized RedirectConsumer getOrCreateRedirectConsumer(String resource) {
        logger.info("getOrCreateRedirectConsumer {}", resource);
        if (unaryConsumerMap.get(resource) == null) {
            RedirectConsumer redirectConsumer =
                    new RedirectConsumer(consumerIdIndex.get(), resource);
            unaryConsumerMap.putIfAbsent(resource, redirectConsumer);
            return (RedirectConsumer) unaryConsumerMap.get(resource);
        } else {
            return (RedirectConsumer) unaryConsumerMap.get(resource);
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

    public void onComplete(String transferId) {
        this.unaryConsumerMap.remove(transferId);
        logger.info("remove consumer {}", transferId);
    }

    private void checkAndClean() {
    }

    @Override
    public void init() {




    }

    @Override
    public void start() {

    }

    @Override
    public void destroy() {

    }

    public static class ReportData {


    }


}
