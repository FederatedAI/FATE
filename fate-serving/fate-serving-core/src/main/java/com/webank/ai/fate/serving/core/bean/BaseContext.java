package com.webank.ai.fate.serving.core.bean;

import com.google.common.collect.Maps;
import com.webank.ai.fate.core.bean.ReturnResult;
import com.webank.ai.fate.serving.core.monitor.WatchDog;
import com.webank.ai.fate.serving.core.utils.GetSystemInfo;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;


import java.util.Map;


public class BaseContext<Req ,Resp extends ReturnResult> implements Context<Req,Resp> {


    long   timestamp;

    LoggerPrinter loggerPrinter ;

    public  BaseContext  (LoggerPrinter loggerPrinter){
        this.loggerPrinter =  loggerPrinter;
        timestamp =  System.currentTimeMillis();
    }
    private  BaseContext(LoggerPrinter loggerPrinter,long  timestamp,Map  dataMap){
        this.timestamp = timestamp;
        this.dataMap = dataMap;
        this.loggerPrinter =loggerPrinter;
    }



    private static final Logger LOGGER = LogManager.getLogger(LOGGER_NAME);

    @Override
    public String getActionType() {
        return actionType;
    }

    String actionType;


    Map dataMap = Maps.newHashMap();

    @Override
    public void preProcess() {


        WatchDog.enter(this);

    }

    @Override
    public Object getData(Object key) {
        return null;
    }

    @Override
    public Object getDataOrDefault(Object  key,Object defaultValue) {
       return  dataMap.getOrDefault(  key, defaultValue);
    }


    @Override
    public void putData(Object key, Object data) {

        dataMap.put(key,data);

    }

    @Override
    public String getCaseId() {
        if(dataMap.get(Dict.CASEID)!=null){
            return  dataMap.get(Dict.CASEID).toString();
        }
        else {
            return null;
        }
    }

    @Override
    public void setCaseId(String caseId){
        dataMap.put(Dict.CASEID,caseId);
    }

    @Override
    public long getTimeStamp() {
        return timestamp;
    }

    @Override
    public void postProcess(Req  req,Resp  resp) {


        try {

            WatchDog.quit(this);

            if(loggerPrinter!=null){
                loggerPrinter.printLog(this,req,resp);
            }

        }catch(Throwable  e){


        }
    }

    @Override
    public void setActionType(String actionType) {
        this.actionType=  actionType;

    }

    @Override
    public ReturnResult getFederatedResult() {
      return   (ReturnResult) dataMap.get(Dict.FEDERATED_RESULT);
    }

    @Override
    public void setFederatedResult(ReturnResult returnResult) {
        dataMap.put(Dict.FEDERATED_RESULT,returnResult);
    }
    @Override
    public boolean isHitCache() {
      return   (Boolean)dataMap.getOrDefault(Dict.HIT_CACHE,false);
    }

    @Override
    public void hitCache(boolean hitCache) {
        dataMap.put(Dict.HIT_CACHE,hitCache);
    }

    @Override
    public Context subContext() {

        Map newDataMap = Maps.newHashMap(dataMap);

       return  new BaseContext(this.loggerPrinter,this.timestamp,dataMap);
    }

    @Override
    public String getSeqNo() {
        return (String) this.dataMap.getOrDefault(Dict.REQUEST_SEQNO,"");
    }

    @Override
    public long getCostTime() {
        return System.currentTimeMillis()-timestamp;
    }
}
