package com.webank.ai.fate.common.mlmodel.model;

import java.util.Map;

public interface MLModel<B, X, P> {
    int setModelId(String modelId);
    String getModelId();
    int initModel(B modelBuffer);
    Map<String, Object> predict(X inputData, P predictParams);
}
