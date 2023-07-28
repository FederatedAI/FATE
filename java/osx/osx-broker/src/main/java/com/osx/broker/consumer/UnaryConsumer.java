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

import com.osx.broker.ServiceContainer;
import com.osx.broker.queue.TransferQueue;
import com.osx.broker.util.TransferUtil;
import com.osx.core.constant.ActionType;
import com.osx.core.constant.StatusCode;
import com.osx.core.context.FateContext;
import com.osx.core.exceptions.ErrorMessageUtil;
import com.osx.core.exceptions.TransferQueueNotExistException;
import com.osx.core.utils.FlowLogUtil;
import io.grpc.stub.StreamObserver;
import lombok.Data;
import org.ppc.ptp.Osx;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.servlet.http.HttpServletResponse;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ConcurrentLinkedQueue;

import static com.osx.broker.util.TransferUtil.buildResponse;

public class UnaryConsumer extends LocalQueueConsumer {

    Logger logger = LoggerFactory.getLogger(UnaryConsumer.class);
    ConcurrentLinkedQueue<LongPullingHold> longPullingQueue;

    public UnaryConsumer(long consumerId, String transferId) {
        super(consumerId, transferId);
        TransferQueue transferQueue = ServiceContainer.transferQueueManager.getQueue(transferId);
        if (transferQueue != null) {
            transferQueue.registerDestoryCallback(() -> {
                ServiceContainer.consumerManager.onComplete(transferId);
            });
        }
        longPullingQueue = new ConcurrentLinkedQueue<>();
    }

    public int getLongPullingQueueSize() {
        return longPullingQueue.size();
    }

    public LongPullingHold pollLongPullingQueue() {
        return longPullingQueue.poll();
    }

    public void addLongPullingQueue(LongPullingHold longPullingHold) {
        longPullingQueue.add(longPullingHold);
        logger.info("add long pulling queue , queue size {}", longPullingQueue.size());
    }

    public synchronized int answerLongPulling() {
        /*
         * 这里需要改为ack  后才加1  ，要不然这里会丢消息
         */
        int answerCount = 0;
        TransferQueue transferQueue = ServiceContainer.transferQueueManager.getQueue(transferId);
        List<LongPullingHold> reputList = null;
        while (this.longPullingQueue.size() > 0) {
            LongPullingHold longPullingHold = this.longPullingQueue.poll();
            try {
                io.grpc.Context  grpcContext = longPullingHold.getGrpcContext();
                if(grpcContext!=null){
                    if(grpcContext.isCancelled()){
                        logger.error("topic {} consumer grpc context is cancelled",transferId);
                        continue;
                    }
                }
                long current= System.currentTimeMillis();
                long needOffset = longPullingHold.getNeedOffset();
                if(transferQueue==null){
                    // TODO: 2023/7/24  这里需要通知阻塞的客户端,最好是由队列清理时主动通知客户端
                    longPullingHold.throwException(new TransferQueueNotExistException());
                    continue;
                }

                if( longPullingHold.getExpireTimestamp()>0&&current>longPullingHold.getExpireTimestamp()){
                    handleExpire(longPullingHold);
                    continue;
                }

                FateContext context = longPullingHold.getContext();
                context.setActionType(ActionType.LONG_PULLING_ANSWER.getAlias());
                TransferQueue.TransferQueueConsumeResult consumeResult = null;
                if (needOffset <= 0) {
                    long consumeOffset = this.consumeOffset.get();
                    if (this.checkMsgIsArrive(consumeOffset)) {
                        /*
                         *  服务器记录的消费进度小于等于 index，则可以消费
                         */
                        consumeResult = this.consume(context, needOffset);
                    }
                } else {
                    if (this.checkMsgIsArrive(needOffset)) {
                        /*
                         *  client 传入的offset 小于等于index，可以消费
                         */
                        consumeResult = this.consume(context, needOffset);
                    }
                }

                if (consumeResult != null) {
                    if (consumeResult.getMessage() != null && consumeResult.getMessage().getBody() != null)
                        context.setDataSize(consumeResult.getMessage().getBody().length);
                    Osx.Outbound consumeResponse = buildResponse(StatusCode.SUCCESS, "success", consumeResult);
                    answerCount++;
                    longPullingHold.answer(consumeResponse);
                    context.setTopic(transferQueue.getTransferId());
                    context.setReturnCode(StatusCode.SUCCESS);
                    context.setRequestMsgIndex(consumeResult.getRequestIndex());
                    context.setCurrentMsgIndex(consumeResult.getLogicIndexTotal());
                    FlowLogUtil.printFlowLog(context);
                } else {
                    /*
                     * 若没有消息，则放回队列
                     */
                    if (reputList == null)
                        reputList = new ArrayList<>();
                    reputList.add(longPullingHold);
                }
            } catch (Exception e) {
                logger.error("topic {} answer long pulling error ",transferId,e);
                longPullingHold.throwException(e);
            }
        }
        if (reputList != null) {
            this.longPullingQueue.addAll(reputList);
        }
        return answerCount;
    }

    private  void handleExpire(LongPullingHold longPullingHold){
        Osx.Outbound consumeResponse = buildResponse(StatusCode.CONSUME_MSG_TIMEOUT, "CONSUME_MSG_TIMEOUT", null);
        longPullingHold.answer(consumeResponse);
    }

    @Data
    public static class LongPullingHold {
        Logger logger = LoggerFactory.getLogger(LongPullingHold.class);
        FateContext context;
        io.grpc.Context   grpcContext;
        StreamObserver streamObserver;
        HttpServletResponse httpServletResponse;
        long expireTimestamp;
        long needOffset;

        public  void  answer(Osx.Outbound  consumeResponse){
            logger.info("============ answer long pulling========");

            if(streamObserver!=null) {

                streamObserver.onNext(consumeResponse);
                streamObserver.onCompleted();
            }else if(httpServletResponse!=null){
                TransferUtil.writeHttpRespose(httpServletResponse,consumeResponse.getCode(),consumeResponse.getMessage(),consumeResponse.getPayload()!=null?consumeResponse.getPayload().toByteArray():null);
            }
        }
        public  void  throwException(Throwable  throwable){
            logger.info("============ answer throw exception========");
            try {
                if (streamObserver != null) {
                    streamObserver.onError(ErrorMessageUtil.toGrpcRuntimeException(throwable));
                    streamObserver.onCompleted();
                } else if (httpServletResponse != null) {

                    // TODO: 2023/7/24  http 处理未添加
                    //  TransferUtil.writeHttpRespose(httpServletResponse,consumeResponse.getCode(),consumeResponse.getMessage(),consumeResponse.getPayload()!=null?consumeResponse.getPayload().toByteArray():null);
                }
            }catch(Exception e){
                logger.error("send error back to consumer , occury error",e);
            }
        }


    }

}
