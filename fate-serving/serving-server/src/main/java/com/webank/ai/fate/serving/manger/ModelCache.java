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

package com.webank.ai.fate.serving.manger;

import com.google.common.cache.CacheBuilder;
import com.google.common.cache.CacheLoader;
import com.google.common.cache.LoadingCache;
import com.webank.ai.fate.core.utils.Configuration;
import com.webank.ai.fate.serving.federatedml.PipelineTask;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;

public class ModelCache {
    private static final Logger LOGGER = LogManager.getLogger();
    private static LoadingCache<String, PipelineTask> modelCache;

    static {
        modelCache = CacheBuilder.newBuilder()
                .expireAfterAccess(Configuration.getPropertyInt("modelCacheAccessTTL"), TimeUnit.HOURS)
                .maximumSize(Configuration.getPropertyInt("modelCacheMaxSize"))
                .build(new CacheLoader<String, PipelineTask>() {
                    @Override
                    public PipelineTask load(String s) throws Exception {
                        return loadModel(s);
                    }
                });
    }

    public static PipelineTask loadModel(String modelKey) {
        String[] modelKeyFields = ModelUtils.splitModelKey(modelKey);
        return ModelUtils.loadModel(modelKeyFields[0], modelKeyFields[1]);
    }

    public static PipelineTask get(String modelKey) {
        try {
            return modelCache.get(modelKey);
        } catch (ExecutionException ex) {
            LOGGER.error(ex);
            return null;
        }
    }

    public static void put(String modelKey, PipelineTask model) {
        modelCache.put(modelKey, model);
    }

    public static long getSize() {
        return modelCache.size();
    }
}
