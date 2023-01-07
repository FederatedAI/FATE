/*
 * Copyright 2019 The FATE Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.osx.core.context;
import com.google.common.collect.Maps;
import com.google.common.util.concurrent.ListenableFuture;
import com.osx.core.constant.Dict;
import com.osx.core.router.RouterInfo;
import com.osx.core.utils.FlowLogPrinter;
import com.osx.core.utils.FlowLogUtil;
import org.apache.commons.lang3.StringUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Map;


public class Context {

    static final String LOGGER_NAME = "flow";

    private static final Logger logger = LoggerFactory.getLogger(LOGGER_NAME);

    protected long timestamp = System.currentTimeMillis();
    protected boolean needAssembleException = false;
    protected String actionType;
    protected String sessionId;
    protected boolean needPrintFlowLog = true;
    protected Long dataSize;
    protected Map dataMap = Maps.newHashMap();
    long costTime;
    String resourceName;
    Throwable t;
    FlowLogPrinter flowLogPrinter = FlowLogUtil::printFlowLog;

    public Long getDataSize() {
        return dataSize;
    }

    public void setDataSize(long dataSize) {
        this.dataSize = dataSize;
    }

    public String getTopic() {
        if (dataMap.get(Dict.TOPIC) != null)
            return dataMap.get(Dict.TOPIC).toString();
        return null;
    }

    public void setTopic(String topic) {
        this.dataMap.put(Dict.TOPIC, topic);
    }

    public String getInstanceId() {
        return (String) dataMap.get(Dict.INSTANCE_ID);

    }

    public void setInstanceId(String instanceId) {
        this.dataMap.put(Dict.INSTANCE_ID, instanceId);
    }

    public Throwable getException() {
        return t;
    }

    public void setException(Throwable t) {
        this.t = t;
    }

    public String getSessionId() {
        return this.sessionId;
    }

    public void setSessionId(String sessionId) {
        this.sessionId = sessionId;
    }

    public String getActionType() {
        return actionType;
    }

    public void setActionType(String actionType) {
        this.actionType = actionType;
    }

    public Object getData(Object key) {
        return dataMap.get(key);
    }

    public Object getDataOrDefault(Object key, Object defaultValue) {
        return dataMap.getOrDefault(key, defaultValue);
    }

    public void putData(Object key, Object data) {
        dataMap.put(key, data);
    }

    public String getCaseId() {
        if (dataMap.get(Dict.CASEID) != null) {
            return dataMap.get(Dict.CASEID).toString();
        } else {
            return null;
        }
    }

    public void setCaseId(String caseId) {
        dataMap.put(Dict.CASEID, caseId);
    }

    public long getTimeStamp() {
        return timestamp;
    }

    public Context subContext() {
        Map newDataMap = Maps.newHashMap(dataMap);
        return new Context(this.timestamp, newDataMap);
    }

    public boolean needPrintFlowLog() {
        return needPrintFlowLog;
    }

    public void setNeedPrintFlowLog(boolean needPrintFlowLog) {
        this.needPrintFlowLog = needPrintFlowLog;
    }

    public Long getRequestMsgIndex() {
        return (Long) this.dataMap.get(Dict.REQUEST_INDEX);
    }

    public void setRequestMsgIndex(Long index) {
        this.dataMap.put(Dict.REQUEST_INDEX, index);
    }

    public Long getCurrentMsgIndex() {
        return (Long) this.dataMap.get(Dict.CURRENT_INDEX);
    }

    public void setCurrentMsgIndex(Long index) {
        this.dataMap.put(Dict.CURRENT_INDEX, index);
    }

    public long getCostTime() {
        return costTime;
    }

    public String getSrcPartyId() {
        return (String) dataMap.get(Dict.SOURCE_PARTY_ID);
    }

    public void setSrcPartyId(String guestAppId) {
        dataMap.put(Dict.SOURCE_PARTY_ID, guestAppId);
    }

    public String getDesPartyId() {
        return (String) dataMap.get(Dict.DES_PARTY_ID);
    }

    public void setDesPartyId(String hostAppid) {
        dataMap.put(Dict.DES_PARTY_ID, hostAppid);
    }

    public  void setSrcComponent(String srcComponent){
        dataMap.put(Dict.SOURCE_COMPONENT,srcComponent);
    }

    public  String getSrcComponent(){
        return (String)dataMap.get(Dict.SOURCE_COMPONENT);
    }

    public  void setDesComponent(String desComponent){
        dataMap.put(Dict.DES_COMPONENT,desComponent);
    }

    public  String getDesComponent(){
        return   (String)dataMap.get(Dict.DES_COMPONENT);
    }



    public RouterInfo getRouterInfo() {
        return (RouterInfo) dataMap.get(Dict.ROUTER_INFO);
    }

    public void setRouterInfo(RouterInfo routerInfo) {
        dataMap.put(Dict.ROUTER_INFO, routerInfo);
    }

    public Object getResultData() {
        return dataMap.get(Dict.RESULT_DATA);
    }

    public void setResultData(Object resultData) {
        dataMap.put(Dict.RESULT_DATA, resultData);
    }

    public String getReturnCode() {
        return (String) dataMap.get(Dict.RETURN_CODE);
    }

    public void setReturnCode(String returnCode) {
        dataMap.put(Dict.RETURN_CODE, returnCode);
    }


    public String getReturnMsg() {
        return (String) dataMap.get(Dict.RET_MSG);
    }

    public void setReturnMsg(String returnMsg) {
        dataMap.put(Dict.RET_MSG, returnMsg);
    }


    public long getDownstreamCost() {
        if (dataMap.get(Dict.DOWN_STREAM_COST) != null) {

            return (long) dataMap.get(Dict.DOWN_STREAM_COST);
        }
        return 0;
    }

    public void setDownstreamCost(long downstreamCost) {
        dataMap.put(Dict.DOWN_STREAM_COST, downstreamCost);
    }

    public long getDownstreamBegin() {
        return dataMap.get(Dict.DOWN_STREAM_BEGIN) != null ? (long) dataMap.get(Dict.DOWN_STREAM_BEGIN) : 0;
    }

    public void setDownstreamBegin(long downstreamBegin) {
        dataMap.put(Dict.DOWN_STREAM_BEGIN, downstreamBegin);
    }

    public String getSourceIp() {
        return (String) dataMap.get(Dict.SOURCE_IP);
    }

    public void setSourceIp(String sourceIp) {
        dataMap.put(Dict.SOURCE_IP, sourceIp);
    }

    public String getServiceName() {
        return (String) dataMap.get(Dict.SERVICE_NAME);
    }

    public void setServiceName(String serviceName) {
        dataMap.put(Dict.SERVICE_NAME, serviceName);
    }

    public String getCallName() {
        return (String) dataMap.get(Dict.CALL_NAME);
    }

    public void setCallName(String callName) {
        dataMap.put(Dict.CALL_NAME, callName);
    }

    public void setRemoteFuture(ListenableFuture future) {
        this.dataMap.put(Dict.FUTURE, future);
    }

    public String getResourceName() {
        if (StringUtils.isNotEmpty(resourceName)) {
            return resourceName;
        } else {
            resourceName = "I_" + (StringUtils.isNotEmpty(this.getActionType()) ? this.getActionType() : this.getServiceName());
        }
        return resourceName;
    }

    public boolean needAssembleException() {
        return needAssembleException;
    }

    public FlowLogPrinter getFlowLogPrinter() {
        return flowLogPrinter;
    }

    public Context setFlowLogPrinter(FlowLogPrinter flowLogPrinter) {
        this.flowLogPrinter = flowLogPrinter;
        return this;
    }

    public void printFlowLog() {
        if (needPrintFlowLog) {
            flowLogPrinter.print(this);
        }
    }


}
