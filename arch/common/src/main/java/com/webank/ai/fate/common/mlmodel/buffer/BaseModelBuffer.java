package com.webank.ai.fate.common.mlmodel.buffer;

import java.util.ArrayList;

public abstract class BaseModelBuffer<K, V> implements ModelBuffer<K, V> {
    public abstract void setMetaField(K name, V value);
    public abstract void setParamField(K name, V value);
    public abstract V getMetaField(K name);
    public abstract V getParamField(K name);
    public abstract ArrayList<byte[]> serialize();
    public abstract int deserialize(byte[] metaStream, byte[] paramStream);
}
