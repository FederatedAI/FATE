package com.webank.ai.fate.common.mlmodel.model;

public interface MLModel<B, X, P, R> {
    B export_model();
    int init_model(B modelBuffer);
    R predict(X inputData, P predictParams);
}
