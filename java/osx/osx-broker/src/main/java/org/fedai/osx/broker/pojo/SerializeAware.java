package org.fedai.osx.broker.pojo;

import org.fedai.osx.core.utils.JsonUtil;

import java.nio.charset.StandardCharsets;

public interface SerializeAware {
    public  byte[] serialize();
}
