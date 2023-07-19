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

import com.osx.api.router.RouterInfo;
import com.osx.core.constant.TransferStatus;


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
