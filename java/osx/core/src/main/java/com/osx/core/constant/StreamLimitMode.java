package com.osx.core.constant;

public enum StreamLimitMode {
    //不使用限流
    NOLIMIT,
    //集群限流
    CLUSTER,
    //本地限流
    LOCAL
}
