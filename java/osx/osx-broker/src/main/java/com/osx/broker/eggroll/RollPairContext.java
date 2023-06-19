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
package com.osx.broker.eggroll;

import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.osx.core.constant.Dict;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class RollPairContext {

    public static ExecutorService executor = Executors.newCachedThreadPool();

    ;
    Logger logger = LoggerFactory.getLogger(RollPairContext.class);
    private String sessionId;
    private ErSession erSession;
    private ErSessionMeta sessionMeta;
    //  val EGGROLL_ROLLPAIR_DEFAULT_STORE_TYPE = ErConfKey("eggroll.rollpair.default.store.type", "ROLLPAIR_LMDB")
    private String defaultStoreType = "ROLLPAIR_LMDB";
    private String defaultSerdesType = "PICKLE";

    public RollPairContext(ErSession erSession) {
        this.erSession = erSession;
    }

    public RollPair load(String namespace, String name, Map<String, String> options) {


//        def load(namespace: String, name: String, options: Map[String,String] = Map()): RollPair = {
//                // TODO:1: use snake case universally?
//                val defaultStoreTypeValue = defaultStoreType.split("_")(1)
//                val storeType = options.getOrElse(StringConstants.STORE_TYPE, options.getOrElse(StringConstants.STORE_TYPE_SNAKECASE, defaultStoreTypeValue))
//                val totalPartitions = options.getOrElse(StringConstants.TOTAL_PARTITIONS, options.getOrElse(StringConstants.TOTAL_PARTITIONS_SNAKECASE, "1")).toInt
//                val store = ErStore(storeLocator = ErStoreLocator(
//                namespace = namespace,
//                name = name,
//                storeType = storeType,
//                totalPartitions = totalPartitions,
//                partitioner = options.getOrElse(StringConstants.PARTITIONER, PartitionerTypes.BYTESTRING_HASH),
//                serdes = options.getOrElse(StringConstants.SERDES, defaultSerdesType)
//        ), options = options.asJava)
//                val loaded = session.clusterManagerClient.getOrCreateStore(store)
//                new RollPair(loaded, this)
//        }

        String defaultStoreTypeValue = defaultStoreType.split("_")[1];

        String storeType = options.getOrDefault(Dict.STORE_TYPE, options.getOrDefault(Dict.STORE_TYPE_SNAKECASE, defaultStoreTypeValue));
        int totalPartitions = Integer.parseInt(options.getOrDefault(Dict.TOTAL_PARTITIONS, options.getOrDefault(Dict.TOTAL_PARTITIONS_SNAKECASE, "1")));

        ErStoreLocator erStoreLocator = new ErStoreLocator(namespace, name, Dict.EMPTY, storeType, totalPartitions,
                options.getOrDefault(Dict.PARTITIONER, PartitionerTypes.BYTESTRING_HASH.name()),
                options.getOrDefault(Dict.SERDES, defaultSerdesType));
        ErStore store = new ErStore(erStoreLocator, Lists.newArrayList(), options);

       // logger.info("===================ppppppp==={}", store);
        ErStore loaded = erSession.clusterManagerClient.getOrCreateStore(store);

       // logger.info("loaded  erStore {}", loaded);

        return new RollPair(loaded, this, Maps.newHashMap());

    }

    public String getSessionId() {
        return sessionId;
    }

    public void setSessionId(String sessionId) {
        this.sessionId = sessionId;
    }

    public ErSessionMeta getSessionMeta() {
        return sessionMeta;
    }

    public void setSessionMeta(ErSessionMeta sessionMeta) {
        this.sessionMeta = sessionMeta;
    }

    public String getDefaultStoreType() {
        return defaultStoreType;
    }

    public void setDefaultStoreType(String defaultStoreType) {
        this.defaultStoreType = defaultStoreType;
    }

    public String getDefaultSerdesType() {
        return defaultSerdesType;
    }

    public void setDefaultSerdesType(String defaultSerdesType) {
        this.defaultSerdesType = defaultSerdesType;
    }

    public ErSession getErSession() {
        return erSession;
    }

    public void setErSession(ErSession erSession) {
        this.erSession = erSession;
    }


//
//    public RollPair load();

}


//class RollPairContext(val session: ErSession,
//                      defaultStoreType: String = RollPairConfKeys.EGGROLL_ROLLPAIR_DEFAULT_STORE_TYPE.get(),
//        defaultSerdesType: String = SerdesTypes.PICKLE) extends Logging {

//private val sessionId = session.sessionId
//private val sessionMeta = session.sessionMeta
//
//        def routeToEgg(partition: ErPartition): ErProcessor = session.routeToEgg(partition)
//
//        def load(namespace: String, name: String, options: Map[String,String] = Map()): RollPair = {
//        // TODO:1: use snake case universally?
//        val defaultStoreTypeValue = defaultStoreType.split("_")(1)
//        val storeType = options.getOrElse(StringConstants.STORE_TYPE, options.getOrElse(StringConstants.STORE_TYPE_SNAKECASE, defaultStoreTypeValue))
//        val totalPartitions = options.getOrElse(StringConstants.TOTAL_PARTITIONS, options.getOrElse(StringConstants.TOTAL_PARTITIONS_SNAKECASE, "1")).toInt
//        val store = ErStore(storeLocator = ErStoreLocator(
//        namespace = namespace,
//        name = name,
//        storeType = storeType,
//        totalPartitions = totalPartitions,
//        partitioner = options.getOrElse(StringConstants.PARTITIONER, PartitionerTypes.BYTESTRING_HASH),
//        serdes = options.getOrElse(StringConstants.SERDES, defaultSerdesType)
//        ), options = options.asJava)
//        val loaded = session.clusterManagerClient.getOrCreateStore(store)
//        new RollPair(loaded, this)
//        }
//
//        // todo:1: partitioner factory depending on string, and mod partition number
//        def partitioner(k: Array[Byte], n: Int): Int = {
//        // Integer.MIN_VALUE  ==  Math.abs(Integer.MIN_VALUE)
//        hashKey(k) % n
//        }
//        def hashKey(k: Array[Byte]): Int = {
//        var h = Math.abs(ByteString.copyFrom(k).hashCode())
//        if (h == Integer.MIN_VALUE) {
//        h = 1
//        }
//        h
//        }
//        }