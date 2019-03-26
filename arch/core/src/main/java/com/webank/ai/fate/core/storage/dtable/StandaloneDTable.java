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

import com.webank.ai.fate.core.constant.StatusCode;
import org.fusesource.lmdbjni.*;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import java.nio.file.Paths;

public class StandaloneDTable implements DTable{
    private static final Logger LOGGER = LogManager.getLogger();
    private String dataDir = Paths.get(System.getProperty("user.dir"), "data").toString();
    private Database db;

    @Override
    public void init(String name, String nameSpace, int partition) {
        String path = Paths.get(this.dataDir, "LMDB", nameSpace, name, Integer.toString(partition)).toString();
        Env env = new Env(path);
        this.db = env.openDatabase();
    }


    @Override
    public byte[] get(String key){
        return db.get(key.getBytes());
    }

    @Override
    public void put(String key, byte[] value) {
        this.db.put(key.getBytes(), value);
    }
}
