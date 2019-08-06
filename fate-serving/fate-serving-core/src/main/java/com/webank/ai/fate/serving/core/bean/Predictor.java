package com.webank.ai.fate.serving.core.bean;

public interface Predictor <Req_I,Req_P,Res>{

    public Res predict(Context context , Req_I inputData, Req_P predictParams) ;
}
