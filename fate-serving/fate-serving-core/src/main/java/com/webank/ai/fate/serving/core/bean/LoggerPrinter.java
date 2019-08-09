package com.webank.ai.fate.serving.core.bean;

public interface LoggerPrinter<Req,Rsp>{

    void  printLog(Context context, Req req, Rsp rsp);
}
