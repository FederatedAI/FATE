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
import com.google.common.collect.Maps;
import com.google.protobuf.ByteString;
import com.webank.ai.eggroll.core.api.grpc.client.crud.StorageMetaClient;
import com.webank.ai.eggroll.core.constant.RuntimeConstants;
import com.webank.ai.eggroll.core.error.exception.StorageNotExistsException;
import com.webank.ai.eggroll.core.io.StoreInfo;
import com.webank.ai.eggroll.core.utils.ErrorUtils;
import com.webank.ai.eggroll.core.utils.TypeConversionUtils;
import com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Dtable;
import com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Fragment;
import com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Node;
import com.webank.ai.eggroll.framework.roll.strategy.DispatchPolicy;
import com.webank.ai.eggroll.framework.roll.strategy.Dispatcher;
import com.webank.ai.eggroll.framework.roll.util.RollServerUtils;
import org.apache.commons.lang3.StringUtils;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Scope;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;

@Component
@Scope("prototype")
public class DefaultDispatcher implements Dispatcher {
    private static final Logger LOGGER = LogManager.getLogger();
    @Autowired
    private StorageMetaClient storageMetaClient;
    @Autowired
    private TypeConversionUtils typeConversionUtils;
    @Autowired
    private ErrorUtils errorUtils;
    @Autowired
    private RollServerUtils rollServerUtils;
    private DispatchPolicy dispatchPolicy;
    private LoadingCache<StoreInfo, Node> storeInfoNodeCache;
    private LoadingCache<Long, Map<Integer, Fragment>> tableIdToMappedFragments;
    private LoadingCache<StoreInfo, Dtable> storeInfoDtableCache;

    public DefaultDispatcher(DispatchPolicy dispatchPolicy) {
        this.dispatchPolicy = dispatchPolicy;
    }

    @PostConstruct
    public void init() {
        storageMetaClient.init(rollServerUtils.getMetaServiceEndpoint());
        tableIdToMappedFragments = CacheBuilder.newBuilder()
                .expireAfterAccess(RuntimeConstants.DEFAULT_WAIT_TIME, RuntimeConstants.DEFAULT_TIMEUNIT)
                .maximumSize(100)
                .recordStats()
                .weakKeys()
                .weakValues()
                .build(new CacheLoader<Long, Map<Integer, Fragment>>() {
                    @Override
                    public Map<Integer, Fragment> load(Long key) throws Exception {
                        Map<Integer, Fragment> result = Maps.newHashMap();
                        List<Fragment> fragments = storageMetaClient.getFragmentsByTableId(key);

                        // todo: consider master / backup scenario
                        for (Fragment fragment : fragments) {
                            result.put(fragment.getFragmentOrder(), fragment);
                        }

                        return result;
                    }
                });

        storeInfoNodeCache = CacheBuilder.newBuilder()
                .expireAfterAccess(RuntimeConstants.DEFAULT_WAIT_TIME, RuntimeConstants.DEFAULT_TIMEUNIT)
                .maximumSize(100)
                .recordStats()
                .weakKeys()
                .weakValues()
                .build(new CacheLoader<StoreInfo, Node>() {
                    @Override
                    public Node load(StoreInfo key) throws Exception {
                        if (StringUtils.isAnyBlank(key.getNameSpace(), key.getTableName()) || key.getFragment() == null) {
                            throw new StorageNotExistsException(key);
                        }

                        StoreInfo noFragmentStoreInfo = StoreInfo.copy(key);
                        noFragmentStoreInfo.setFragment(null);

                        Dtable dtable = storeInfoDtableCache.get(noFragmentStoreInfo);
                        if (dtable == null || dtable.getTableId() == null) {
                            throw new StorageNotExistsException(key);
                        }
                        Map<Integer, Fragment> fragmentMap = tableIdToMappedFragments.get(dtable.getTableId());
                        Fragment targetFragment = fragmentMap.get(key.getFragment());

                        if (targetFragment == null) {
                            throw new StorageNotExistsException(key);
                        }

                        Node result = storageMetaClient.getNodeByFragmentId(targetFragment.getFragmentId());

                        return result;
                    }
                });

        storeInfoDtableCache = CacheBuilder.newBuilder()
                .expireAfterAccess(RuntimeConstants.DEFAULT_WAIT_TIME, RuntimeConstants.DEFAULT_TIMEUNIT)
                .maximumSize(100)
                .recordStats()
                .weakKeys()
                .weakValues()
                .build(new CacheLoader<StoreInfo, Dtable>() {
                    @Override
                    public Dtable load(StoreInfo key) throws Exception {
                        if (StringUtils.isAnyBlank(key.getNameSpace(), key.getTableName())) {
                            throw new StorageNotExistsException(key);
                        }

                        Dtable result = storageMetaClient.getTable(key.getNameSpace(), key.getTableName());

                        return result;
                    }
                });
    }

    @Override
    public Node dispatch(StoreInfo storeInfo, ByteString dataKey) {
        Node result = null;
        StoreInfo duplicate = StoreInfo.copy(storeInfo);

        try {
            Dtable dtable = storeInfoDtableCache.get(duplicate);
            if (dtable == null) {
                throw new StorageNotExistsException(duplicate);
            }

            int dispatchResult = dispatchPolicy.executePolicy(dtable.getTotalFragments(), dataKey);
            duplicate.setFragment(dispatchResult);

            result = storeInfoNodeCache.get(duplicate);
        } catch (ExecutionException e) {
            LOGGER.error(errorUtils.getStackTrace(e));
        }


        return result;
    }
}
