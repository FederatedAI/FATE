package com.osx.broker.consumer;

import com.osx.broker.ServiceContainer;
import com.osx.broker.queue.TransferQueue;
import com.osx.core.constant.ActionType;
import com.osx.core.constant.StatusCode;
import com.osx.core.context.Context;
import com.osx.core.utils.FlowLogUtil;
import io.grpc.stub.StreamObserver;
import lombok.Data;
import org.ppc.ptp.Osx;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ConcurrentLinkedQueue;

import static com.osx.broker.util.TransferUtil.buildResponse;

public class UnaryConsumer extends LocalQueueConsumer {

    Logger logger = LoggerFactory.getLogger(UnaryConsumer.class);
    ConcurrentLinkedQueue<LongPullingHold> longPullingQueue;

    public UnaryConsumer(long consumerId, String transferId

    ) {
        super(consumerId, transferId);
        TransferQueue transferQueue = ServiceContainer.transferQueueManager.getQueue(transferId);
        if (transferQueue != null) {
            transferQueue.registeDestoryCallback(() -> {
                ServiceContainer.consumerManager.onComplete(transferId);
            });
        }

        longPullingQueue = new ConcurrentLinkedQueue();
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
            try {
                long indexFileOffset = transferQueue.getIndexQueue().getLogicOffset().get();
                LongPullingHold longPullingHold = this.longPullingQueue.poll();
                //StreamObserver  streamObserver = longPullingHold.getStreamObserver();
                long needOffset = longPullingHold.getNeedOffset();
                Context context = longPullingHold.getContext();
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

                //logger.info("{} longpulling consume {} return {}",transferId,consumeOffset.get(),message);
                if (consumeResult != null) {
                    if (consumeResult.getMessage() != null && consumeResult.getMessage().getBody() != null)
                        context.setDataSize(consumeResult.getMessage().getBody().length);
                    Osx.Outbound consumeResponse = buildResponse(StatusCode.SUCCESS, "success", consumeResult);
                    answerCount++;
                    longPullingHold.getStreamObserver().onNext(consumeResponse);
                    longPullingHold.getStreamObserver().onCompleted();
//                        if(needOffset<=0){
//                            this.addConsumeCount(1);  改成了由ack自增
//                        }
                    context.setReturnCode(StatusCode.SUCCESS);
                    FlowLogUtil.printFlowLogForConsumer(context);


                } else {
                    /*
                     * 若没有消息，则放回队列
                     */
                    if (reputList == null)
                        reputList = new ArrayList<>();
                    reputList.add(longPullingHold);

                }
            } catch (Exception igore) {
                igore.printStackTrace();
            }
        }
        if (reputList != null) {
            this.longPullingQueue.addAll(reputList);
        }
        return answerCount;
    }

    @Data
    public static class LongPullingHold {
        Context context;
        StreamObserver streamObserver;
        long needOffset;
    }

}
