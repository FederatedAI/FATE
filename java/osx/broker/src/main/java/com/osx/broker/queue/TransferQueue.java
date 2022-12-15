package com.osx.broker.queue;


import com.osx.core.config.MetaInfo;
import com.osx.core.constant.StatusCode;
import com.osx.core.constant.TransferStatus;
import com.osx.core.context.Context;
import com.osx.core.exceptions.TransferQueueInvalidStatusException;
import com.osx.core.queue.TranferQueueInfo;
import com.osx.broker.ServiceContainer;
import com.osx.broker.callback.CompleteCallback;
import com.osx.broker.callback.DestoryCallback;
import com.osx.broker.callback.ErrorCallback;

import com.osx.broker.message.MessageDecoder;
import com.osx.broker.message.MessageExt;
import com.osx.broker.message.MessageExtBrokerInner;
import com.osx.broker.message.SelectMappedBufferResult;
import com.osx.broker.store.IndexQueue;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

public class TransferQueue {
    Logger  logger = LoggerFactory.getLogger(TransferQueue.class);
    String transferId;

    public String getSessionId() {
        return sessionId;
    }

    public void setSessionId(String sessionId) {
        this.sessionId = sessionId;
    }

    public String getSrcPartyId() {
        return srcPartyId;
    }

    public void setSrcPartyId(String srcPartyId) {
        this.srcPartyId = srcPartyId;
    }

    public String getDesPartyId() {
        return desPartyId;
    }

    public void setDesPartyId(String desPartyId) {
        this.desPartyId = desPartyId;
    }

    String sessionId;
    String srcPartyId;
    String desPartyId;

    volatile TransferStatus  transferStatus = TransferStatus.INIT;
    List<ErrorCallback>  errorCallbacks = new ArrayList<>();
    List<CompleteCallback> completeCallbacks = new ArrayList<>();
    List<DestoryCallback>  destoryCallbacks = new ArrayList();
    long createTimestamp;
    long lastStatusChangeTimestamp;
    long lastWriteTimestamp;
    long lastReadTimestamp;
    boolean writeOver= false;



    public IndexQueue getIndexQueue() {
        return indexQueue;
    }

    public void setIndexQueue(IndexQueue indexQueue) {
        this.indexQueue = indexQueue;
    }

    IndexQueue indexQueue;

    protected final AtomicInteger wrotePosition = new AtomicInteger(0);
    TransferQueueManager transferQueueManager;
    public  TransferQueue(String  transferId,TransferQueueManager transferQueueManager ,String path){
        this.transferId = transferId;
        this.transferQueueManager = transferQueueManager;
        this.createTimestamp = System.currentTimeMillis();
        this.lastStatusChangeTimestamp = this.createTimestamp;
        this.lastWriteTimestamp = this.createTimestamp;
        this.indexQueue = new IndexQueue(transferId,path, MetaInfo.PROPERTY_INDEX_MAP_FILE_SIZE);

    }

    public synchronized  PutMessageResult putMessage(final MessageExtBrokerInner msg) {
        if(transferStatus== TransferStatus.TRANSFERING) {
            this.lastWriteTimestamp = System.currentTimeMillis();
            PutMessageResult  putMessageResult = ServiceContainer.messageStore.putMessage(msg);
            if(putMessageResult.isOk()){
                long  beginWriteOffset = putMessageResult.getAppendMessageResult().getWroteOffset();
                int   size = putMessageResult.getAppendMessageResult().getWroteBytes();
                logger.info("store begin offset {},size {}",beginWriteOffset,size);
                putMessageResult.setMsgLogicOffset(  indexQueue.putMessagePositionInfoWrapper(beginWriteOffset,size));
            }else{
                throw new RuntimeException();
            }
            return putMessageResult;
        }else{
            throw  new TransferQueueInvalidStatusException( "invalid queue status : "+transferStatus);
        }
    }

    public static  class TransferQueueConsumeResult {
        public String getCode() {
            return code;
        }

        public void setCode(String code) {
            this.code = code;
        }

        public TransferQueueConsumeResult(String code,
                                          SelectMappedBufferResult selectMappedBufferResult,
                                          long requestIndex,
                                          long logicIndex){
            this.code = code;
            this.selectMappedBufferResult = selectMappedBufferResult;
            this.requestIndex = requestIndex;
            this.logicIndexTotal = logicIndex;
        }

        public SelectMappedBufferResult getSelectMappedBufferResult() {
            return selectMappedBufferResult;
        }

        public void setSelectMappedBufferResult(SelectMappedBufferResult selectMappedBufferResult) {
            this.selectMappedBufferResult = selectMappedBufferResult;
        }

        public long getRequestIndex() {
            return requestIndex;
        }

        public void setRequestIndex(long requestIndex) {
            this.requestIndex = requestIndex;
        }

        SelectMappedBufferResult  selectMappedBufferResult;
        long  requestIndex;

        public long getLogicIndexTotal() {
            return logicIndexTotal;
        }

        public void setLogicIndexTotal(long logicIndexTotal) {
            this.logicIndexTotal = logicIndexTotal;
        }

        long  logicIndexTotal;

        String code="-1";

        public MessageExt getMessage() {
            return message;
        }

        public void setMessage(MessageExt message) {
            this.message = message;
        }

        MessageExt  message;
    }

    public TransferQueueConsumeResult consumeOneMessage(Context context, long requestIndex)  {
        TransferQueueConsumeResult  transferQueueConsumeResult;

        if(transferStatus==TransferStatus.TRANSFERING) {
            this.lastReadTimestamp = System.currentTimeMillis();
            long logicIndex = indexQueue.getLogicOffset().get();
            context.setRequestMsgIndex(requestIndex);
            context.setCurrentMsgIndex(logicIndex);
            if(requestIndex<= logicIndex) {
                SelectMappedBufferResult indexBufferResult = this.indexQueue.getIndexBuffer(requestIndex);
                if(indexBufferResult!=null) {
                    long pyOffset = indexBufferResult.getByteBuffer().getLong();
                    SelectMappedBufferResult msgBufferResult = ServiceContainer.messageStore.consumeOneMessage(pyOffset);
                    transferQueueConsumeResult = new  TransferQueueConsumeResult(StatusCode.SUCCESS,msgBufferResult,requestIndex,logicIndex);
                    MessageExt message = MessageDecoder.decode(transferQueueConsumeResult.getSelectMappedBufferResult().getByteBuffer());
                    transferQueueConsumeResult.setMessage(message);
                }else{
                    transferQueueConsumeResult = new  TransferQueueConsumeResult(StatusCode.INVALID_INDEXFILE_DETAIL,null,requestIndex,logicIndex);
                }
            }else{
                transferQueueConsumeResult= new TransferQueueConsumeResult(StatusCode.CONSUME_NO_MESSAGE,null,requestIndex,logicIndex);
            }
        }else{
            throw new  TransferQueueInvalidStatusException("transfer queue invalid status : "+transferStatus);
        }
        return  transferQueueConsumeResult;
    }




    public synchronized void  destory(){
        logger.info("try to destory transfer queue {} ",transferId);
        this.indexQueue.destroy();
        logger.info("destroy index file");
        destoryCallbacks.forEach(destoryCallback -> {
            try{
                destoryCallback.callback();
            }catch(Exception e){
                logger.error("destory call back error",e);
            }
        });
    }

    public long getCreateTimestamp() {
        return createTimestamp;
    }

    public void setCreateTimestamp(long createTimestamp) {
        this.createTimestamp = createTimestamp;
    }

    public synchronized void  onCompeleted(){
        if(transferStatus==TransferStatus.TRANSFERING) {

            transferStatus = TransferStatus.FINISH;
        }
        completeCallbacks.forEach(completeCallback -> {
            try{
                completeCallback.callback();
            }catch (Exception e){

            }
        });
    }

    public synchronized void  onError(Throwable  throwable){
        logger.error("transfer queue {} productor error",transferId,throwable);
        if(transferStatus==TransferStatus.TRANSFERING) {
            transferStatus = TransferStatus.ERROR;
        }
        errorCallbacks.forEach(errorCallback -> {
            try {
                errorCallback.callback(throwable);
            }catch(Exception e){
                logger.error("error call back ",e);
            }
        });
    }

    public synchronized void registeErrorCallback(ErrorCallback  errorCallback){
        if(transferStatus==TransferStatus.TRANSFERING) {
            errorCallbacks.add(errorCallback);
        }else{
            throw  new TransferQueueInvalidStatusException("status is "+transferStatus);
        }
    }

    public synchronized  void registeDestoryCallback(DestoryCallback  destoryCallback){
        if(transferStatus==TransferStatus.TRANSFERING)
            destoryCallbacks.add(destoryCallback);
        else
            throw  new TransferQueueInvalidStatusException("status is "+transferStatus);
    }

    public TransferStatus getTransferStatus() {
        return transferStatus;
    }

//    public void setTransferStatus(TransferStatus transferStatus) {
//        this.transferStatus = transferStatus;
//    }
    public AtomicInteger getWrotePosition() {
        return wrotePosition;
    }

    public boolean isWriteOver() {
        return writeOver;
    }

    public void setWriteOver(boolean writeOver) {
        this.writeOver = writeOver;
    }

    public String getTransferId() {
        return transferId;
    }

    public void setTransferId(String transferId) {
        this.transferId = transferId;
    }

    public synchronized void  start(){
        logger.info("transfer queue {} start ",transferId);
        if(this.transferStatus==TransferStatus.INIT){
            this.transferStatus = TransferStatus.TRANSFERING;
        }
    }
    public long getLastReadTimestamp() {
        return lastReadTimestamp;
    }

    public void setLastReadTimestamp(long lastReadTimestamp) {
        this.lastReadTimestamp = lastReadTimestamp;
    }

    public long getLastWriteTimestamp() {
        return lastWriteTimestamp;
    }

    public void setLastWriteTimestamp(long lastWriteTimestamp) {
        this.lastWriteTimestamp = lastWriteTimestamp;
    }

    public TranferQueueInfo getTransferQueueInfo(){
        TranferQueueInfo   transferQueueInfo = new TranferQueueInfo();
        transferQueueInfo.setTransferId(transferId);
        transferQueueInfo.setCreateTimestamp(createTimestamp);
        transferQueueInfo.setLastReadTimestamp(lastReadTimestamp);
        transferQueueInfo.setLastWriteTimestamp(lastWriteTimestamp);
        transferQueueInfo.setTransferStatus(transferStatus);
        transferQueueInfo.setLogicOffset(indexQueue.getLogicOffset().get());
        return  transferQueueInfo;
    }


}
