package org.fedai.osx.broker.queue;

import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.fedai.osx.broker.callback.CompleteCallback;
import org.fedai.osx.broker.callback.DestoryCallback;
import org.fedai.osx.broker.callback.ErrorCallback;
import org.fedai.osx.broker.callback.MsgEventCallback;
import org.fedai.osx.broker.constants.MessageFlag;
import org.fedai.osx.broker.consumer.ConsumerManager;
import org.fedai.osx.core.constant.TransferStatus;
import org.fedai.osx.core.context.OsxContext;
import org.fedai.osx.core.exceptions.TransferQueueInvalidStatusException;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

@Data
@Slf4j
public class AbstractQueue  {
    protected final AtomicInteger wrotePosition = new AtomicInteger(0);
    volatile TransferStatus transferStatus = TransferStatus.INIT;
    String transferId;
    String sessionId;
    String srcPartyId;
    String desPartyId;
    long createTimestamp;
    long lastStatusChangeTimestamp;
    long lastWriteTimestamp;
    long lastReadTimestamp;
    TransferQueueManager transferQueueManager;
    ConsumerManager consumerManager;

    List<ErrorCallback> errorCallbacks = new ArrayList<>();
    List<CompleteCallback> completeCallbacks = new ArrayList<>();
    List<DestoryCallback> destoryCallbacks = new ArrayList<>();
    List<MsgEventCallback> msgCallbacks = new ArrayList<>();

    public synchronized void putMessage(OsxContext context, Object data , MessageFlag messageFlag, String msgCode)  {

    }

    public synchronized TransferQueueConsumeResult consumeOneMessage(OsxContext context, long requestIndex){
        return null;
    }


    public synchronized void registerErrorCallback(ErrorCallback errorCallback) {
        if (transferStatus == TransferStatus.TRANSFERING) {
            errorCallbacks.add(errorCallback);
        } else {
            throw new TransferQueueInvalidStatusException("status is " + transferStatus);
        }
    }

    public synchronized void registerDestoryCallback(DestoryCallback destoryCallback) {
        if (transferStatus == TransferStatus.TRANSFERING)
            destoryCallbacks.add(destoryCallback);
        else
            throw new TransferQueueInvalidStatusException("status is " + transferStatus);
    }

    public synchronized void registerMsgCallback(List<MsgEventCallback> msgCallbacks) {
        if (transferStatus == TransferStatus.TRANSFERING) {
            this.msgCallbacks.addAll(msgCallbacks);
        } else
            throw new TransferQueueInvalidStatusException("status is " + transferStatus);
    }

    public synchronized void onCompeleted() {
        if (transferStatus == TransferStatus.TRANSFERING) {
            transferStatus = TransferStatus.FINISH;
        }
        completeCallbacks.forEach(completeCallback -> {
            try {
                completeCallback.callback();
            } catch (Exception e) {
                log.error("complete call back error", e);
            }
        });
    }

    public synchronized void onError(Throwable throwable) {
        log.error("transfer queue {} productor error", transferId, throwable);
        if (transferStatus == TransferStatus.TRANSFERING) {
            transferStatus = TransferStatus.ERROR;
        }
        errorCallbacks.forEach(errorCallback -> {
            try {
                errorCallback.callback(throwable);
            } catch (Exception e) {
                log.error("error call back ", e);
            }
        });
    }

    public synchronized void destory() {
        log.info("try to destory transfer queue {} ", transferId);
        destoryCallbacks.forEach(destoryCallback -> {
            try {
                destoryCallback.callback();
            } catch (Exception e) {
                log.error("topic {} destory call back execute error", transferId, e);
            }
        });
    }

    public synchronized void start() {
        log.info("topic {} start ", transferId);
        if (this.transferStatus == TransferStatus.INIT) {
            this.transferStatus = TransferStatus.TRANSFERING;
        }
    }


}
