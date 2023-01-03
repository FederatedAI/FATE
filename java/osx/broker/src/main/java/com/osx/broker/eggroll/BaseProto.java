package com.osx.broker.eggroll;


import com.osx.core.utils.JsonUtil;

public abstract class BaseProto<T> {

    abstract T toProto();

    public String toString() {
        return JsonUtil.object2Json(this);
    }
}
