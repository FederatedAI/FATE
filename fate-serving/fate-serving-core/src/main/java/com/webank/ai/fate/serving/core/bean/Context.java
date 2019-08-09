package com.webank.ai.fate.serving.core.bean;

import com.webank.ai.fate.core.bean.ReturnResult;

public interface Context <Req,Resp>{


    static final   String  LOGGER_NAME =  "flow";
    public void  preProcess();
    public Object  getData(Object  key);
    public Object  getDataOrDefault(Object  key,Object defaultValue);
    public void   putData(Object  key,Object data);
    public String getCaseId();
    public void setCaseId(String caseId);
    public long  getTimeStamp();
    public default void  postProcess(Req req,Resp resp){};
    public void  setActionType(String actionType);
    public ReturnResult  getFederatedResult();
    public void  setFederatedResult(ReturnResult  returnResult);
    public boolean  isHitCache();
    public void  hitCache(boolean   hitCache);
    public Context  subContext();
    public String getActionType();
    public String getSeqNo();
    public long  getCostTime();


}
