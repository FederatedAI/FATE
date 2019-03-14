package com.webank.ai.fate.common.mlmodel.model;


public abstract class BaseModel<ModelBuffer, X, P, R> implements MLModel<ModelBuffer, X, P, R>{
    public BaseModel(){
    }
    public abstract ModelBuffer export_model();
    public abstract int init_model(ModelBuffer modelBuffer);
    public abstract R predict(X inputData, P predictParams);
}
