package com.osx.broker.consumer;

import com.osx.core.constant.TransferStatus;
import com.osx.core.router.RouterInfo;

import java.util.concurrent.atomic.AtomicBoolean;

public class RedirectConsumer extends UnaryConsumer {

    RouterInfo routerInfo;
    TransferStatus transferStatus;
    AtomicBoolean isWorking = new AtomicBoolean(false);


    public RedirectConsumer(long consumerId, String transferId
    ) {
        super(consumerId, transferId);
        transferStatus = TransferStatus.TRANSFERING;
    }

    public RouterInfo getRouterInfo() {
        return routerInfo;
    }

    public void setRouterInfo(RouterInfo routerInfo) {
        this.routerInfo = routerInfo;
    }

    @Override
    public TransferStatus getTransferStatus() {
        return transferStatus;
    }

    @Override
    public void setTransferStatus(TransferStatus transferStatus) {
        this.transferStatus = transferStatus;
    }

    public boolean getIsWorking() {
        return isWorking.get();
    }

    public boolean setIsWorking(boolean pre, boolean update) {
        return isWorking.compareAndSet(pre, update);
    }


}
