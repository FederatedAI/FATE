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

package com.webank.ai.fate.core.storage.dtable;

import com.webank.ai.fate.core.utils.Configuration;
import org.fusesource.lmdbjni.*;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;

public class StandaloneDTable implements DTable {
    private static final Logger LOGGER = LogManager.getLogger();
    //private String dataDir = Paths.get(System.getProperty("user.dir"), "data").toString();
    private String dataDir = Configuration.getProperty("standaloneStoragePath");
    private Database db;
    private Env env;

    public StandaloneDTable(String name, String namespace, int partition) {
        String path = Paths.get(this.dataDir, "LMDB", namespace, name, Integer.toString(0)).toString();
        LOGGER.info(path);
        this.env = new Env(path);
        this.db = this.env.openDatabase();
    }

    @Override
    public byte[] get(String key) {
        return db.get(key.getBytes());
    }

    @Override
    public void put(String key, byte[] value) {
        this.db.put(key.getBytes(), value);
    }

    @Override
    public Map<String, byte[]> collect() {
        Map<String, byte[]> kvData = new HashMap<>();
        for (Entry next : db.iterate(this.env.createReadTransaction()).iterable()) {
            kvData.put(new String(next.getKey()), next.getValue());
        }
        return kvData;
    }
}
