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

package com.webank.ai.eggroll.core.io;

import com.webank.ai.eggroll.api.storage.StorageBasic;
import com.webank.ai.eggroll.core.constant.MetaConstants;
import com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Dtable;
import lombok.Builder;
import lombok.Data;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

@Data
@Builder
// todo: make fragment optional
public class StoreInfo {
    private static final Logger LOGGER = LogManager.getLogger();
    final String type;
    final String nameSpace;
    final String tableName;
    Integer fragment;

    public static StoreInfo fromGrpcContext() {
        StoreInfoBuilder builder = StoreInfo.builder()
                .nameSpace(MetaConstants.NAME_SPACE.asContextKey().get())
                .tableName(MetaConstants.TABLE_NAME.asContextKey().get())
                .type(MetaConstants.STORE_TYPE.asContextKey().get());
        try {
            builder.fragment(Integer.parseInt(MetaConstants.FRAGMENT.asContextKey().get()));
        } catch (Exception e) {
            LOGGER.warn("no fragment in store info");
        }

        return builder.build();
    }

/*
    public static StoreInfo fromDTable(Processor.DTable dTable) {
        return StoreInfo.builder().fragment(dTable.getFragment())
                .nameSpace(dTable.getNamespace())
                .tableName(dTable.getTable())
                .type(dTable.getType()).build();
    }
*/

    public static StoreInfo fromStorageLocator(StorageBasic.StorageLocator storageLocator) {
        return StoreInfo.builder()
                .nameSpace(storageLocator.getNamespace())
                .tableName(storageLocator.getName())
                .fragment(storageLocator.getFragment())
                .type(storageLocator.getType().name())
                .build();
    }

    public static StoreInfo fromDtable(Dtable dtable) {
        return StoreInfo.builder()
                .nameSpace(dtable.getNamespace())
                .tableName(dtable.getTableName())
                .type(dtable.getTableType())
                .build();
    }

    public static StoreInfo copy(StoreInfo another) {
        return StoreInfo.builder()
                .type(another.getType())
                .nameSpace(another.getNameSpace())
                .tableName(another.getTableName())
                .fragment(another.getFragment())
                .build();
    }
}
