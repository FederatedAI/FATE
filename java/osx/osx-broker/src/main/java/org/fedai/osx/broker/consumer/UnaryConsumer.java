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
package org.fedai.osx.broker.consumer;

import com.google.gson.Gson;
import io.grpc.stub.StreamObserver;
import lombok.Data;

import org.fedai.osx.broker.pojo.ConsumerResponse;
import org.fedai.osx.broker.queue.TransferQueue;
import org.fedai.osx.broker.queue.TransferQueueConsumeResult;
import org.fedai.osx.broker.queue.TransferQueueManager;
import org.fedai.osx.broker.util.TransferUtil;
import org.fedai.osx.core.constant.ActionType;
import org.fedai.osx.core.constant.Dict;
import org.fedai.osx.core.constant.StatusCode;
import org.fedai.osx.core.context.OsxContext;
import org.fedai.osx.core.exceptions.ErrorMessageUtil;
import org.fedai.osx.core.exceptions.TransferQueueNotExistException;
import org.fedai.osx.core.utils.FlowLogUtil;
import org.fedai.osx.core.utils.JsonUtil;
import org.ppc.ptp.Osx;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.servlet.AsyncContext;
import javax.servlet.http.HttpServletResponse;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Base64;
import java.util.List;
import java.util.concurrent.ConcurrentLinkedQueue;

public class UnaryConsumer extends LocalQueueConsumer {

    Logger logger = LoggerFactory.getLogger(UnaryConsumer.class);
    ConcurrentLinkedQueue<LongPullingHold> longPullingQueue;
    TransferQueueManager  transferQueueManager;
    ConsumerManager consumerManager;
    static Base64.Encoder base64Encoder = Base64.getEncoder();

    public UnaryConsumer(TransferQueueManager  transferQueueManager,ConsumerManager consumerManager,long consumerId,String sessionId, String topic) {
        super(transferQueueManager,consumerId,sessionId, topic);
        this.transferQueueManager = transferQueueManager;
        this.consumerManager = consumerManager;

        TransferQueue transferQueue = (TransferQueue) transferQueueManager.getQueue(sessionId,topic);
        if (transferQueue != null) {
            transferQueue.registerDestoryCallback(() -> {
                String indexKey = TransferQueueManager.assembleTopic(sessionId,topic);
               consumerManager.onComplete(indexKey);
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
        TransferQueue transferQueue = (TransferQueue) transferQueueManager.getQueue(sessionId,topic);
        List<LongPullingHold> reputList = null;
        while (this.longPullingQueue.size() > 0) {
            LongPullingHold longPullingHold = this.longPullingQueue.poll();
            try {
                io.grpc.Context  grpcContext = longPullingHold.getGrpcContext();
                if(grpcContext!=null){
                    if(grpcContext.isCancelled()){
                        logger.error("session {} topic {} consumer grpc context is cancelled",sessionId,topic);
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
                OsxContext context = longPullingHold.getContext();
                context.setActionType(ActionType.LONG_PULLING_ANSWER.name());
                TransferQueueConsumeResult consumeResult = null;
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
                    answerCount++;
                    longPullingHold.answer(consumeResult,StatusCode.PTP_SUCCESS, Dict.SUCCESS);
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
                logger.error("session {} topic {} answer long pulling error ",sessionId,topic,e);
                longPullingHold.throwException(e);
            }
        }
        if (reputList != null) {
            this.longPullingQueue.addAll(reputList);
        }
        return answerCount;
    }

    private  void handleExpire(LongPullingHold longPullingHold){
        longPullingHold.answer(null,StatusCode.PTP_TIME_OUT,"CONSUME_MSG_TIMEOUT");
    }

    @Data
    public static class LongPullingHold {
        Logger logger = LoggerFactory.getLogger(LongPullingHold.class);
        OsxContext context;
        io.grpc.Context   grpcContext;
        StreamObserver streamObserver;
        AsyncContext  asyncContext;
        long expireTimestamp;
        long needOffset;
        public  void  answer(TransferQueueConsumeResult  consumeResult,String  statusCode,String message){
            if(streamObserver!=null) {
                Osx.TransportOutbound consumeResponse = TransferUtil.buildTransportOutbound(statusCode, message, consumeResult);
                streamObserver.onNext(consumeResponse);
                streamObserver.onCompleted();
            }else if(asyncContext!=null){
                byte[]  content = null;
                if(consumeResult!=null&&consumeResult.getMessage()!=null){
                    content = consumeResult.getMessage().getBody();
                }
                ConsumerResponse  consumerResponse = new ConsumerResponse();
                consumerResponse.setCode(statusCode);
                consumerResponse.setMsg(message);
                if(content!=null)
                    consumerResponse.setPayload(content);
                String returnContent = JsonUtil.object2Json(consumerResponse);

                TransferUtil.writeHttpRespose(asyncContext.getResponse(),statusCode,message, returnContent.getBytes(StandardCharsets.UTF_8));
                asyncContext.complete();


            }
        }


        public  void  throwException(Throwable  throwable){
            try {
                if (streamObserver != null) {
                    streamObserver.onError(ErrorMessageUtil.toGrpcRuntimeException(throwable));
                    streamObserver.onCompleted();
                } else if (asyncContext != null) {

                    // TODO: 2023/7/24  http 处理未添加
                    //  TransferUtil.writeHttpRespose(httpServletResponse,consumeResponse.getCode(),consumeResponse.getMessage(),consumeResponse.getPayload()!=null?consumeResponse.getPayload().toByteArray():null);
                }
            }catch(Exception e){
                logger.error("send error back to consumer , occury error",e);
            }
        }


    }

}
