package com.webank.ai.fate.common.mlmodel.buffer;

import java.util.ArrayList;

public interface ModelBuffer<K, V> {
    void setMetaField(K name, V value);
    void setParamField(K name, V value);
    V getMetaField(K name);
    V getParamField(K name);
    ArrayList<byte[]> serialize();
    int deserialize(byte[] metaStream, byte[] paramStream);
}
