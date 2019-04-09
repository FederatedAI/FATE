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

import com.webank.ai.fate.core.storage.kv.BaseKVPool;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

import com.webank.ai.fate.serving.federatedml.PipelineTask;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

public class ModelPool extends BaseKVPool<String, PipelineTask> {
    private static final Logger LOGGER = LogManager.getLogger();
    private static final Map<String, PipelineTask> pool = new HashMap<>();
    private final ReadWriteLock lock = new ReentrantReadWriteLock();
    private final Lock readLock = lock.readLock();
    private final Lock writeLock = lock.writeLock();

    @Override
    public void put(String key, PipelineTask value){
        this.writeLock.lock();
        try {
            pool.put(key, value);
        }
        catch (Exception ex) {
            LOGGER.error(ex);
        }
        finally {
            this.writeLock.unlock();
        }
    }

    @Override
    public void putIfAbsent(String key, PipelineTask value){
        this.writeLock.lock();
        try {
            pool.putIfAbsent(key, value);
        }
        catch (Exception ex) {
            LOGGER.error(ex);
        }
        finally {
            this.writeLock.unlock();
        }
    }

    @Override
    public void putAll(Map<String, PipelineTask> kv){
        this.writeLock.lock();
        try {
            pool.putAll(kv);
        }
        catch (Exception ex) {
            LOGGER.error(ex);
        }
        finally {
            this.writeLock.unlock();
        }
    }

    @Override
    public PipelineTask get(String key){
        this.readLock.lock();
        PipelineTask pipelineTask = pool.get(key);
        this.readLock.unlock();
        return pipelineTask;
    }

    public ArrayList<String> keys(){
        if (pool.size() > 0){
            return new ArrayList<String>(pool.keySet());
        }
        else{
            return null;
        }
    }
}
