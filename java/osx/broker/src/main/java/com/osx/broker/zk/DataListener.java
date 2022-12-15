package com.osx.broker.zk;

public interface DataListener {

    void dataChanged(String path, Object value, EventType eventType);
}
