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

package com.webank.ai.eggroll.framework.roll.strategy.impl;

import com.google.common.cache.CacheBuilder;
import com.google.common.cache.CacheLoader;
import com.google.common.cache.LoadingCache;
import com.google.protobuf.ByteString;
import com.webank.ai.eggroll.core.api.grpc.client.crud.StorageMetaClient;
import com.webank.ai.eggroll.core.constant.RuntimeConstants;
import com.webank.ai.eggroll.core.error.exception.StorageNotExistsException;
import com.webank.ai.eggroll.core.io.StoreInfo;
import com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Dtable;
import com.webank.ai.eggroll.framework.roll.strategy.DispatchPolicy;
import com.webank.ai.eggroll.framework.roll.util.RollServerUtils;
import org.apache.commons.lang3.StringUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Scope;
import org.springframework.stereotype.Service;

import javax.annotation.PostConstruct;
import java.util.concurrent.ExecutionException;

@Service("dispatchPolicy")
@Scope("prototype")
public class DefaultModDispatchPolicy implements DispatchPolicy {
    @Autowired
    private StorageMetaClient storageMetaClient;
    @Autowired
    private RollServerUtils rollServerUtils;

    private LoadingCache<StoreInfo, Integer> storeInfoToFragmentCountLoadingCache;

    @PostConstruct
    public void init() {
        storageMetaClient.init(rollServerUtils.getMetaServiceEndpoint());
        storeInfoToFragmentCountLoadingCache = CacheBuilder.newBuilder()
                .weakKeys()
                .weakValues()
                .maximumSize(100)
                .expireAfterAccess(RuntimeConstants.DEFAULT_WAIT_TIME, RuntimeConstants.DEFAULT_TIMEUNIT)
                .recordStats()
                .build(new CacheLoader<StoreInfo, Integer>() {
                    @Override
                    public Integer load(StoreInfo key) throws Exception {
                        if (StringUtils.isAnyBlank(key.getNameSpace(), key.getTableName())) {
                            throw new StorageNotExistsException(key);
                        }
                        Dtable dtable = storageMetaClient.getTable(key);
                        if (dtable == null) {
                            throw new StorageNotExistsException(key);
                        }

                        return dtable.getTotalFragments();
                    }
                });
    }

    @Override
    public int executePolicy(int total, ByteString key) {
        int hashValue = key.hashCode();
        if (hashValue == Integer.MIN_VALUE) {
            hashValue = 0;
        }

        return Math.abs(hashValue) % total;
    }

    @Override
    public int executePolicy(StoreInfo storeInfo, ByteString key) {
        try {
            return executePolicy(storeInfoToFragmentCountLoadingCache.get(storeInfo), key);
        } catch (ExecutionException e) {
            throw new RuntimeException(e);
        }
    }
}
