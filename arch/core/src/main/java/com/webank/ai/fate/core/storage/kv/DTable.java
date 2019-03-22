package com.webank.ai.fate.core.storage.kv;

public interface DTable {
    void init(String name, String nameSpace, int partition);
    byte[] get(String key);
}
