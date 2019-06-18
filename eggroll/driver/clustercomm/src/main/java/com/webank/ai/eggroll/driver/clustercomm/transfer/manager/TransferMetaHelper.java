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

package com.webank.ai.eggroll.driver.clustercomm.transfer.manager;

import com.google.common.base.Preconditions;
import com.webank.ai.eggroll.api.driver.clustercomm.ClusterComm;
import com.webank.ai.eggroll.core.factory.ReturnStatusFactory;
import com.webank.ai.eggroll.core.utils.ErrorUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

/**
 * All updates to TransferMeta in Manager must be operated through this helper
 */
@Component
public class TransferMetaHelper {
    @Autowired
    private TransferMetaManager transferMetaManager;
    @Autowired
    private ReturnStatusFactory returnStatusFactory;
    @Autowired
    private ErrorUtils errorUtils;

    public void create(ClusterComm.TransferMeta transferMeta) {
        transferMetaManager.create(transferMeta);
    }

    public void onError(ClusterComm.TransferMeta transferMeta, int code, String message) {
        Preconditions.checkNotNull(transferMeta, "transferMeta cannot be null");

        ClusterComm.TransferMeta proposed = transferMeta.toBuilder()
                .setTransferStatus(ClusterComm.TransferStatus.ERROR)
                .setReturnStatus(returnStatusFactory.create(code, message))
                .build();

        boolean result = transferMetaManager.update(proposed);

        if (!result) {
            throw new IllegalStateException("failed to update transferMeta. should not get here");
        }
    }

    public void onError(ClusterComm.TransferMeta transferMeta, int code, Throwable throwable) {
        onError(transferMeta, code, errorUtils.getStackTrace(throwable));
    }

    public ClusterComm.TransferMeta get(ClusterComm.TransferMeta transferMeta) {
        return transferMetaManager.get(transferMeta);
    }

/*    public ClusterComm.TransferMeta get(String transferMetaId) {
        return transferMetaManager.get(transferMetaId);
    }*/

    public ClusterComm.TransferMeta onInit(ClusterComm.TransferMeta transferMeta) {
        return updateStatus(transferMeta, ClusterComm.TransferStatus.INITIALIZING);
    }

    public ClusterComm.TransferMeta onProcess(ClusterComm.TransferMeta transferMeta) {
        return updateStatus(transferMeta, ClusterComm.TransferStatus.PROCESSING);
    }

    public ClusterComm.TransferMeta onComplete(ClusterComm.TransferMeta transferMeta) {
        return updateStatus(transferMeta, ClusterComm.TransferStatus.COMPLETE);
    }

    public ClusterComm.TransferMeta updateStatus(ClusterComm.TransferMeta transferMeta, ClusterComm.TransferStatus transferStatus) {
        Preconditions.checkNotNull(transferMeta, "input transferMeta cannot be null");
        ClusterComm.TransferMeta result = null;
        ClusterComm.TransferMeta managedTransferMeta = transferMetaManager.get(transferMeta);

        ClusterComm.TransferMeta.Builder builder = ClusterComm.TransferMeta.newBuilder().mergeFrom(managedTransferMeta);
        builder.setTransferStatus(transferStatus);

        ClusterComm.TransferMeta newStatus = builder.build();
        boolean updateResult = transferMetaManager.update(newStatus);

        if (updateResult) {
            result = newStatus;
        }

        return result;
    }

    public boolean update(ClusterComm.TransferMeta transferMeta) {
        boolean result = false;
        if (transferMetaManager.get(transferMeta) != null) {
            result = transferMetaManager.update(transferMeta);
        }
        return result;
    }
}
