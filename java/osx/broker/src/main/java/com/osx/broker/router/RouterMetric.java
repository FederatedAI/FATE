package com.osx.broker.router;






import com.osx.core.utils.JsonUtil;

import java.util.concurrent.atomic.AtomicLong;

public class RouterMetric {

    long  lastCheckTimestamp;

    public long getLastCheckTimestamp() {
        return lastCheckTimestamp;
    }

    public void setLastCheckTimestamp(long lastCheckTimestamp) {
        this.lastCheckTimestamp = lastCheckTimestamp;
    }

    public AtomicLong getSourceReceiveBytesCount() {
        return sourceReceiveBytesCount;
    }

    public void setSourceReceiveBytesCount(AtomicLong sourceReceiveBytesCount) {
        this.sourceReceiveBytesCount = sourceReceiveBytesCount;
    }

    public AtomicLong getSourceSendBytesCount() {
        return sourceSendBytesCount;
    }

    public void setSourceSendBytesCount(AtomicLong sourceSendBytesCount) {
        this.sourceSendBytesCount = sourceSendBytesCount;
    }

    public AtomicLong getSinkReceiveBytesCount() {
        return sinkReceiveBytesCount;
    }

    public void setSinkReceiveBytesCount(AtomicLong sinkReceiveBytesCount) {
        this.sinkReceiveBytesCount = sinkReceiveBytesCount;
    }

    public AtomicLong getSinkSendBytesCount() {
        return sinkSendBytesCount;
    }

    public void setSinkSendBytesCount(AtomicLong sinkSendBytesCount) {
        this.sinkSendBytesCount = sinkSendBytesCount;
    }

    public long getLastUpstreamBytesCount() {
        return lastUpstreamBytesCount;
    }

    public void setLastUpstreamBytesCount(long lastUpstreamBytesCount) {
        this.lastUpstreamBytesCount = lastUpstreamBytesCount;
    }

    public long getLastDownstreamBytesCount() {
        return lastDownstreamBytesCount;
    }

    public void setLastDownstreamBytesCount(long lastDownstreamBytesCount) {
        this.lastDownstreamBytesCount = lastDownstreamBytesCount;
    }

    AtomicLong sourceReceiveBytesCount=new AtomicLong(0);

    AtomicLong sourceSendBytesCount = new AtomicLong(0);

    AtomicLong sinkReceiveBytesCount = new AtomicLong(0);

    AtomicLong sinkSendBytesCount = new AtomicLong(0);

    long  lastUpstreamBytesCount;

    long  lastDownstreamBytesCount;

    public  long  addSourceReceive(int  size){
        return sourceReceiveBytesCount.addAndGet(size);
    }

    public  long   addSourceSend(int size){
        return  sourceSendBytesCount.addAndGet(size);
    }


    public  long  addSinkReceive(int  size){
        return sinkReceiveBytesCount.addAndGet(size);
    }

    public  long   addSinkSend(int size){
        return  sinkSendBytesCount.addAndGet(size);
    }



    public  String toString(){
        return JsonUtil.object2Json(this);
    }

    public  static  void  main(String[] args){
        RouterMetric   routerMetric = new  RouterMetric();
        System.err.println("xxxxxxxxxxxxxxxxxxxxx"+routerMetric);
    }

}
