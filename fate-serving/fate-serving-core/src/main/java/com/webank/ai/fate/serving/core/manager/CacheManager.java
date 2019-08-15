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

package com.webank.ai.fate.serving.core.manager;

import com.google.common.base.Charsets;
import com.google.common.cache.Cache;
import com.google.common.cache.CacheBuilder;
import com.google.common.hash.Hashing;
import com.webank.ai.fate.core.bean.FederatedParty;
import com.webank.ai.fate.core.bean.FederatedRoles;
import com.webank.ai.fate.core.bean.ReturnResult;
import com.webank.ai.fate.core.utils.Configuration;
import com.webank.ai.fate.core.utils.FederatedUtils;
import com.webank.ai.fate.core.utils.ObjectTransform;
import com.webank.ai.fate.serving.core.bean.CacheValueConfig;
import com.webank.ai.fate.serving.core.bean.Context;
import org.apache.commons.lang3.StringUtils;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import redis.clients.jedis.Jedis;
import redis.clients.jedis.JedisPool;
import redis.clients.jedis.JedisPoolConfig;
import redis.clients.jedis.Pipeline;

import java.util.*;
import java.util.concurrent.TimeUnit;

public class CacheManager {
    private static final Logger LOGGER = LogManager.getLogger();
    private static JedisPool jedisPool;
    private static Cache<String, ReturnResult> inferenceResultCache;
    private static Cache<String, ReturnResult> remoteModelInferenceResultCache;
    private static int[] remoteModelInferenceResultCacheDBIndex;
    private static int[] inferenceResultCacheDBIndex;
    private static int externalRemoteModelInferenceResultCacheTTL;
    private static int externalInferenceResultCacheTTL;
    private static Set<Integer> canCacheRetcode;

    private enum CacheType {
        INFERENCE_RESULT,
        REMOTE_MODEL_INFERENCE_RESULT
    }

    static {
        remoteModelInferenceResultCache = CacheBuilder.newBuilder()
                .expireAfterAccess(Configuration.getPropertyInt("remoteModelInferenceResultCacheTTL"), TimeUnit.SECONDS)
                .maximumSize(Configuration.getPropertyInt("remoteModelInferenceResultCacheMaxSize"))
                .build();

        inferenceResultCache = CacheBuilder.newBuilder()
                .expireAfterAccess(Configuration.getPropertyInt("inferenceResultCacheTTL"), TimeUnit.SECONDS)
                .maximumSize(Configuration.getPropertyInt("inferenceResultCacheCacheMaxSize"))
                .build();

        JedisPoolConfig jedisPoolConfig = new JedisPoolConfig();
        jedisPoolConfig.setMaxTotal(Configuration.getPropertyInt("redis.maxTotal"));
        jedisPoolConfig.setMaxIdle(Configuration.getPropertyInt("redis.maxIdle"));
        jedisPool = new JedisPool(jedisPoolConfig,
                Configuration.getProperty("redis.ip"),
                Configuration.getPropertyInt("redis.port"),
                Configuration.getPropertyInt("redis.timeout"),
                Configuration.getProperty("redis.password"));
        inferenceResultCacheDBIndex = initializeCacheDBIndex(Configuration.getProperty("external.inferenceResultCacheDBIndex"));
        externalInferenceResultCacheTTL = Configuration.getPropertyInt("external.inferenceResultCacheTTL");
        remoteModelInferenceResultCacheDBIndex = initializeCacheDBIndex(Configuration.getProperty("external.remoteModelInferenceResultCacheDBIndex"));
        externalRemoteModelInferenceResultCacheTTL = Configuration.getPropertyInt("external.remoteModelInferenceResultCacheTTL");
        canCacheRetcode = initializeCanCacheRetcode();
    }

    public static void putInferenceResultCache(Context context , String partyId, String caseid, ReturnResult returnResult) {

        long  beginTime =System.currentTimeMillis();
        try {

            String inferenceResultCacheKey = generateInferenceResultCacheKey(partyId, caseid);
            boolean putCacheSuccess = putIntoCache(inferenceResultCacheKey, CacheType.INFERENCE_RESULT, returnResult);
            if (putCacheSuccess) {
                LOGGER.info("Put {} inference result into cache", inferenceResultCacheKey);
            }
        }finally {
            long  end =System.currentTimeMillis();
            LOGGER.info("caseid {} putInferenceResultCache cost {}",context.getCaseId(),end - beginTime);

        }


    }

    public static ReturnResult getInferenceResultCache(String partyId, String caseid) {
        String inferenceResultCacheKey = generateInferenceResultCacheKey(partyId, caseid);
        ReturnResult returnResult = getFromCache(inferenceResultCacheKey, CacheType.INFERENCE_RESULT);
        if (returnResult != null) {
            LOGGER.info("Get {} inference result from cache.", inferenceResultCacheKey);
        }
        return returnResult;
    }

    public static void putRemoteModelInferenceResult(FederatedParty remoteParty, FederatedRoles federatedRoles, Map<String, Object> featureIds, ReturnResult returnResult) {
        if (! Boolean.parseBoolean(Configuration.getProperty("remoteModelInferenceResultCacheSwitch"))){
            return;
        }
        String remoteModelInferenceResultCacheKey = generateRemoteModelInferenceResultCacheKey(remoteParty, federatedRoles, featureIds);
        boolean putCacheSuccess = putIntoCache(remoteModelInferenceResultCacheKey, CacheType.REMOTE_MODEL_INFERENCE_RESULT, returnResult);
        if (putCacheSuccess) {
            LOGGER.info("Put {} remote model inference result into cache.", remoteModelInferenceResultCacheKey);
        }
    }

    public static ReturnResult getRemoteModelInferenceResult(FederatedParty remoteParty, FederatedRoles federatedRoles, Map<String, Object> featureIds) {
        if (! Boolean.parseBoolean(Configuration.getProperty("remoteModelInferenceResultCacheSwitch"))){
            return null;
        }
        String remoteModelInferenceResultCacheKey = generateRemoteModelInferenceResultCacheKey(remoteParty, federatedRoles, featureIds);
        ReturnResult returnResult = getFromCache(remoteModelInferenceResultCacheKey, CacheType.REMOTE_MODEL_INFERENCE_RESULT);
        if (returnResult != null) {
            LOGGER.info("Get {} remote model inference result from cache.", remoteModelInferenceResultCacheKey);
        }
        return returnResult;
    }

    private static ReturnResult getFromCache(String cacheKey, CacheType cacheType) {
        CacheValueConfig cacheValueConfig = getCacheValueConfig(cacheKey, cacheType);
        ReturnResult returnResultFromInCache = (ReturnResult) cacheValueConfig.getInProcessCache().getIfPresent(cacheKey);
        if (returnResultFromInCache != null) {
            return returnResultFromInCache;
        }
        ReturnResult returnResultFromExternalCache = getFromRedisCache(cacheKey, cacheValueConfig);
        if (returnResultFromExternalCache != null) {
            cacheValueConfig.getInProcessCache().put(cacheKey, returnResultFromExternalCache);
        }
        return returnResultFromExternalCache;
    }

    private static boolean putIntoCache(String cacheKey, CacheType cacheType, ReturnResult returnResult) {
        CacheValueConfig cacheValueConfig = getCacheValueConfig(cacheKey, cacheType);
        if (canCacheRetcode.contains(returnResult.getRetcode())) {
            cacheValueConfig.getInProcessCache().put(cacheKey, returnResult);
            putIntoRedisCache(cacheKey, cacheValueConfig, returnResult);
            return true;
        } else {
            return false;
        }
    }

    private static ReturnResult getFromRedisCache(String cacheKey, CacheValueConfig cacheValueConfig) {
        try (Jedis jedis = jedisPool.getResource()) {
            jedis.select(cacheValueConfig.getDbIndex());
            String cacheValueString = jedis.get(cacheKey);
            ReturnResult returnResultFromExternalCache = (ReturnResult) ObjectTransform.json2Bean(cacheValueString, ReturnResult.class);
            return returnResultFromExternalCache;
        }

    }

    private static void putIntoRedisCache(String cacheKey, CacheValueConfig cacheValueConfig, ReturnResult returnResult) {
        try (Jedis jedis = jedisPool.getResource()) {
                Pipeline redisPipeline = jedis.pipelined();
                redisPipeline.select(cacheValueConfig.getDbIndex());
                redisPipeline.set(cacheKey, ObjectTransform.bean2Json(returnResult));
                redisPipeline.expire(cacheKey, cacheValueConfig.getTtl());
                redisPipeline.sync();


        }
    }

    private static int[] initializeCacheDBIndex(String config) {
        int[] dbIndexs;
        String[] indexStartEnd = config.split(",");
        if (indexStartEnd.length > 1) {
            int start = Integer.parseInt(indexStartEnd[0]);
            int end = Integer.parseInt(indexStartEnd[1]);
            dbIndexs = new int[end - start + 1];
            for (int i = 0; i < end; i++) {
                dbIndexs[i] = start + i;
            }
        } else {
            dbIndexs = new int[1];
            dbIndexs[0] = Integer.parseInt(indexStartEnd[0]);
        }
        return dbIndexs;
    }

    private static Set<Integer> initializeCanCacheRetcode() {
        Set<Integer> retcodes = new HashSet<>();
        String[] retcodeString = Configuration.getProperty("canCacheRetcode").split(",");
        for (int i = 0; i < retcodeString.length; i++) {
            retcodes.add(Integer.parseInt(retcodeString[i]));
        }
        return retcodes;
    }

    private static CacheValueConfig getCacheValueConfig(String cacheKey, CacheType cacheType) {
        int dbIndex;
        int ttl;
        switch (cacheType) {
            case INFERENCE_RESULT:
                dbIndex = getCacheDBIndex(cacheKey, inferenceResultCacheDBIndex);
                ttl = externalInferenceResultCacheTTL + new Random().nextInt(10);
                return new CacheValueConfig<>(dbIndex, ttl, inferenceResultCache);
            case REMOTE_MODEL_INFERENCE_RESULT:
                dbIndex = getCacheDBIndex(cacheKey, remoteModelInferenceResultCacheDBIndex);
                ttl = externalRemoteModelInferenceResultCacheTTL + new Random().nextInt(100);
                return new CacheValueConfig<>(dbIndex, ttl, remoteModelInferenceResultCache);
            default:
                return null;
        }
    }

    private static int getCacheDBIndex(String cacheKey, int[] dbIndexs) {
        int i = Hashing.murmur3_128().hashString(cacheKey, Charsets.UTF_8).asInt() % dbIndexs.length;
        return dbIndexs[i > 0 ? i : -i];
    }

    private static String generateInferenceResultCacheKey(String partyId, String caseid) {
        return StringUtils.join(Arrays.asList(partyId, caseid), "_");
    }

    private static String generateRemoteModelInferenceResultCacheKey(FederatedParty remoteParty, FederatedRoles federatedRoles, Map<String, Object> featureIds) {
        String remotePartyKey = StringUtils.join(Arrays.asList(remoteParty.getRole(), remoteParty.getPartyId(), FederatedUtils.federatedRolesIdentificationString(federatedRoles)), "#");
        Object[] featureIdKeys = featureIds.keySet().toArray();
        Arrays.sort(featureIdKeys);
        List<String> featureIdItemString = new ArrayList<>();
        for (int i = 0; i < featureIdKeys.length; i++) {
            featureIdItemString.add(StringUtils.join(Arrays.asList(featureIdKeys[i], featureIds.get(featureIdKeys[i])), ":"));
        }
        String featureIdString = StringUtils.join(featureIdItemString, "_");
        String cacheKey;
        cacheKey = StringUtils.join(Arrays.asList(remotePartyKey, featureIdString), "#");
        LOGGER.info(cacheKey);
        return cacheKey;
    }
}
