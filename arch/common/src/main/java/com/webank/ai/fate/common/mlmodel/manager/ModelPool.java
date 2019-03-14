package com.webank.ai.fate.common.mlmodel.manager;

import com.webank.ai.fate.common.mlmodel.model.MLModel;
import com.webank.ai.fate.common.storage.kv.BaseKVPool;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

public class ModelPool extends BaseKVPool<String, MLModel> {
    private static final Logger LOGGER = LogManager.getLogger();
    private static final Map<String, MLModel> pool = new HashMap<>();
    private final ReadWriteLock lock = new ReentrantReadWriteLock();
    private final Lock readLock = lock.readLock();
    private final Lock writeLock = lock.writeLock();

    @Override
    public MLModel put(String key, MLModel value){
        this.writeLock.lock();
        try {
            return pool.put(key, value);
        }
        catch (Exception ex) {
            return null;
        }
        finally {
            this.writeLock.unlock();
        }
    }

    @Override
    public MLModel putIfAbsent(String key, MLModel value){
        this.writeLock.lock();
        try {
            return pool.putIfAbsent(key, value);
        }
        catch (Exception ex) {
            return null;
        }
        finally {
            this.writeLock.unlock();
        }
    }

    @Override
    public void putAll(Map<String, MLModel> kv){
        this.writeLock.lock();
        try {
            pool.putAll(kv);
        }
        catch (Exception ex) {
        }
        finally {
            this.writeLock.unlock();
        }
    }

    @Override
    public MLModel get(String key){
        this.readLock.lock();
        try {
            return pool.get(key);
        }
        catch (Exception ax) {
            return null;
        }
        finally {
            this.readLock.unlock();
        }
    }
}
