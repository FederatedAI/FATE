package com.webank.ai.fate.core.mlmodel.model;

import java.util.Map;

public interface MLModel<B, X, P> {
    void setModelInfo(Map<String, String> modelInfo);
    Map<String, String> getModelInfo();
    int initModel(B modelBuffer);
    Map<String, Object> predict(X inputData, P predictParams);
}
