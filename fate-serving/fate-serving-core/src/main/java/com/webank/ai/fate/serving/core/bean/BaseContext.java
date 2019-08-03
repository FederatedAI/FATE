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

    public  BaseContext  (){
        timestamp =  System.currentTimeMillis();
    }
    private  BaseContext(long  timestamp,Map  dataMap){
        this.timestamp = timestamp;
        this.dataMap = dataMap;
    }


    private static final Logger LOGGER = LogManager.getLogger(LOGGER_NAME);

    String actionType;


    Map dataMap = Maps.newHashMap();

    @Override
    public void preProcess() {


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
            long now = System.currentTimeMillis();
            String reqData = this.dataMap.get(Dict.ORIGIN_REQUEST) != null ? this.dataMap.get(Dict.ORIGIN_REQUEST).toString() : "";
            if (req instanceof Request) {
                LOGGER.info("{}|{}|{}|{}|{}|{}|{}|{}|{}", GetSystemInfo.getLocalIp(), req != null ? ((Request) req).getCaseid() : "NONE", actionType, now - timestamp,
                        resp != null ? resp.getRetcode() : "NONE",WatchDog.get(), req,resp
                );
            }
            if (req instanceof Map) {
                LOGGER.info("{}|{}|{}|{}|{}|{}|{}|{}|{}",
                        GetSystemInfo.getLocalIp(),  req != null ? ((Map) req).get(Dict.CASEID) : "NONE", actionType, now - timestamp,
                        resp != null ? resp.getRetcode() : "NONE",WatchDog.get(), req,resp
                );
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

       return  new BaseContext(this.timestamp,dataMap);
    }
}
