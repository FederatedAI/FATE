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

package com.webank.ai.eggroll.framework.storage.service.manager;

import com.google.common.cache.*;
import com.webank.ai.eggroll.core.io.KeyValueStore;
import com.webank.ai.eggroll.core.io.StoreInfo;
import com.webank.ai.eggroll.core.io.StoreManager;
import com.webank.ai.eggroll.core.model.Bytes;
import com.webank.ai.eggroll.framework.storage.service.model.LMDBStore;
import com.webank.ai.eggroll.framework.storage.service.model.enums.Stores;
import org.apache.commons.io.FileUtils;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.attribute.BasicFileAttributes;
import java.nio.file.attribute.FileTime;
import java.util.Properties;
import java.util.Timer;
import java.util.TimerTask;
import java.util.concurrent.TimeUnit;

public class LMDBStoreManager implements StoreManager<Bytes, byte[]> {
    public static final String LMDB_TEMPORARY = "lmdb_temporary";
    public static final String LMDB = "lmdb";
    public static final long INMEMORY_RENTENTION_TIME = TimeUnit.DAYS.toMillis(2);
    private static Logger LOGGER = LogManager.getLogger(LMDBStoreManager.class);
    private static final RemovalListener<StoreInfo, LMDBStore> REMOVAL_LISTENER = new RemovalListener<StoreInfo, LMDBStore>() {
        @Override
        public void onRemoval(RemovalNotification<StoreInfo, LMDBStore> removalNotification) {
            LMDBStore store = removalNotification.getValue();
            LOGGER.info("Evicting " + store.toString());
            if (store.persistent()) {
                removalNotification.getValue().close();
            } else {
                LOGGER.info("Destroying on removal " + store.toString());
                store.destroy();
            }
        }
    };
    private final Timer timer;
    private final String parentDir;
    private LoadingCache<StoreInfo, LMDBStore> storeCache = CacheBuilder.newBuilder()
            .expireAfterAccess(360, TimeUnit.MINUTES)
            .removalListener(REMOVAL_LISTENER)
            .build(new CacheLoader<StoreInfo, LMDBStore>() {
                @Override
                public LMDBStore load(StoreInfo storeInfo) throws Exception {
                    LMDBStore store = (LMDBStore) Stores.LMDB.create(storeInfo);
                    LOGGER.info("Loading " + store.toString());
                    if (!store.isOpen()) {
                        Properties properties = new Properties();
                        if (storeInfo.getType().equalsIgnoreCase(Stores.IN_MEMORY.name())) {
                            // should config the same as python processor
                            properties.put(LMDBStore.DATA_DIR, Paths.get(parentDir, LMDB_TEMPORARY).toString());
                        } else {
                            properties.put(LMDBStore.DATA_DIR, Paths.get(parentDir, LMDB).toString());
                        }
                        store.init(properties);
                        LOGGER.info("Initiated " + store.toString());
                    }
                    return store;
                }
            });

    public LMDBStoreManager(String parentDir) {
        this.parentDir = parentDir;
        timer = new Timer();
        timer.scheduleAtFixedRate(new TimerTask() {
            @Override
            public void run() {
                cleanRoutine();
            }
        }, 0, TimeUnit.HOURS.toMillis(1));
    }

    @Override
    public KeyValueStore<Bytes, byte[]> getStore(StoreInfo info) {
        return storeCache.getIfPresent(info);
    }

    @Override
    public KeyValueStore<Bytes, byte[]> createIfMissing(StoreInfo info) {
        return storeCache.getUnchecked(info);
    }

    @Override
    public void destroy(StoreInfo info) {
        KeyValueStore<Bytes, byte[]> store = storeCache.getUnchecked(info);
        store.destroy();
        storeCache.invalidate(info);
        LOGGER.info("Destroyed explicitly " + store.toString());
    }

    public void destroy() {
        for (KeyValueStore<Bytes, byte[]> store : storeCache.asMap().values()) {
            store.destroy();
            LOGGER.info("Destroyed in iter" + store.toString());
        }
        storeCache.cleanUp();
    }

    private void cleanRoutine() {
        File tempDirectory = Paths.get(parentDir, LMDB_TEMPORARY).toFile();
        if (tempDirectory == null || !tempDirectory.isDirectory()) {
            return;
        }
        File[] inMemories = tempDirectory.listFiles();
        if (inMemories == null) {
            return;
        }
        for (File dbFile : inMemories) {
            checkRetentionAndRemove(dbFile);
        }
    }

    private void checkRetentionAndRemove(File file) {
        boolean deleted = false;
        try {
            BasicFileAttributes attrs = Files.readAttributes(file.toPath(), BasicFileAttributes.class);
            FileTime fileTime = attrs.lastModifiedTime();
            long current = System.currentTimeMillis();
            long elapsed = current - fileTime.toMillis();
            if (elapsed >= INMEMORY_RENTENTION_TIME) {
                deleted = FileUtils.deleteQuietly(file);
                LOGGER.info("Remove " + file.toPath().toString() + " " + deleted);
            }
        } catch (IOException e) {
            LOGGER.error("Get namespace meta error", e);
        }

        if (!deleted && file.isDirectory()) {
            File[] subFiles = file.listFiles();
            if (null == subFiles) {
                return;
            }
            for (File subFile : subFiles) {
                checkRetentionAndRemove(subFile);
            }
        }
    }
}
