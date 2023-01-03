package com.osx.broker.queue;


import com.osx.core.frame.ServiceThread;

public class TransferQueueMonitorService extends ServiceThread {

    TransferQueueManager transferQueueManager;

    public TransferQueueMonitorService(TransferQueueManager transferQueueManager) {
        this.transferQueueManager = transferQueueManager;
    }

    @Override
    public String getServiceName() {
        return "TransferQueueMonitorService";
    }

    @Override
    public void run() {


    }
}
