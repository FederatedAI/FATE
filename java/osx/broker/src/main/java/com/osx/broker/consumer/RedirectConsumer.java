package com.osx.broker.consumer;

import com.osx.core.router.RouterInfo;
import com.osx.core.constant.TransferStatus;

import java.util.concurrent.atomic.AtomicBoolean;

public class RedirectConsumer extends UnaryConsumer {

    public RouterInfo getRouterInfo() {
        return routerInfo;
    }

    public void setRouterInfo(RouterInfo routerInfo) {
        this.routerInfo = routerInfo;
    }

    RouterInfo  routerInfo;


    @Override
    public TransferStatus getTransferStatus() {
        return transferStatus;
    }

    @Override
    public void setTransferStatus(TransferStatus transferStatus) {
        this.transferStatus = transferStatus;
    }

    TransferStatus  transferStatus ;

    public boolean  getIsWorking() {
        return isWorking.get();
    }

    public boolean  setIsWorking(boolean  pre,boolean update) {
        return isWorking.compareAndSet(pre,update);
    }

    AtomicBoolean  isWorking = new  AtomicBoolean(false);

    public  RedirectConsumer(long consumerId,String transferId
                             ){
        super( consumerId, transferId);
        transferStatus = TransferStatus.TRANSFERING;
    }



}
