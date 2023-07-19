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
import com.osx.api.router.RouterInfo;
import com.osx.core.config.MetaInfo;
import com.osx.core.constant.Dict;
import com.osx.api.constants.Protocol;

import com.osx.core.utils.FlowLogUtil;
import org.apache.commons.lang3.StringUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.util.Map;
import com.osx.api.context.Context;
public class FateContext implements Context{
    protected long timestamp = System.currentTimeMillis();
    protected boolean needAssembleException = false;
    protected String actionType;
    protected String sessionId;
    protected Protocol  protocol;
    protected String  traceId;
    protected String token;
    protected String sourceInstId;
    protected String desInstId;
    protected String techProviderCode;
    protected boolean needPrintFlowLog = true;
    protected boolean needCheckRouterInfo = true;
    protected Long dataSize;

    public Integer getRetryTime() {
        return retryTime;
    }

    public void setRetryTime(Integer retryTime) {
        this.retryTime = retryTime;
    }

    protected Integer retryTime =1;
    protected Map dataMap = Maps.newHashMap();
    long costTime;
    String resourceName;
    String messageFlag;
    String messageCode;

    public String getJobId() {
        return jobId;
    }

    @Override
    public void setJobId(String jobId) {
        this.jobId = jobId;
    }

    String jobId;

    Throwable t;
    public  FateContext(){
    }
    public  FateContext(long  timestamp, Map dataMap){
        timestamp = timestamp;
        this.dataMap =  dataMap;
    }

    public boolean isDestination(){
        if(StringUtils.isNotEmpty(this.getDesPartyId()))
            return MetaInfo.PROPERTY_SELF_PARTY.contains(this.getDesPartyId());
        else
            return false;
    }
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
    public String getTechProviderCode() {
        return techProviderCode;
    }
    public void setTechProviderCode(String techProviderCode) {
        this.techProviderCode = techProviderCode;
    }
    public Protocol getProtocol() {
        return protocol;
    }
    public void setProtocol(Protocol protocol) {
        this.protocol = protocol;
    }
    public String getMessageFlag() {
        return messageFlag;
    }
    public String getMessageCode() {
        return messageCode;
    }
    public void setMessageCode(String messageCode) {
        this.messageCode = messageCode;
    }
    public void setMessageFlag(String messageFlag) {
        this.messageFlag = messageFlag;
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
    public String getTraceId() {
        return traceId;
    }
    public void setTraceId(String traceId) {
        this.traceId = traceId;
    }
    public String getToken() {
        return token;
    }
    public void setToken(String token) {
        this.token = token;
    }
    public boolean isNeedCheckRouterInfo() {
        return needCheckRouterInfo;
    }
    public void setNeedCheckRouterInfo(boolean needCheckRouterInfo) {
        this.needCheckRouterInfo = needCheckRouterInfo;
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

    public FateContext subContext() {
        Map newDataMap = Maps.newHashMap(dataMap);
        return new FateContext(this.timestamp, newDataMap);
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
    public String getSelfPartyId(){
        return (String)  dataMap.get(Dict.PROPERTY_SELF_PARTY_KEY);
    }
    public void setSelfPartyId(String partyId){
        dataMap.put(Dict.PROPERTY_SELF_PARTY_KEY,partyId);
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
    public String toString(){
        StringBuffer stringBuffer = new StringBuffer();
        if (this.getProtocol() != null) {
            stringBuffer.append(this.getProtocol()).append(SPLIT);
        }
        if (this.getActionType() != null) {
            stringBuffer.append(this.getActionType()).append(SPLIT);
        }
//        if(context.getSessionId()!=null){
//            stringBuffer.append("session:").append(context.getSessionId()).append(SPLIT);
//        }
        if (this.getTopic() != null) {
            stringBuffer.append("topic:").append(this.getTopic()).append(SPLIT);
        }

        if (this.getMessageFlag() != null) {
                stringBuffer.append(this.getMessageFlag()).append(SPLIT);
        }
        if (this.getRequestMsgIndex() != null) {
            stringBuffer.append("req-offset:").append(this.getRequestMsgIndex()).append(SPLIT);
        }
        if (this.getData(Dict.CURRENT_INDEX) != null) {
            stringBuffer.append("offset-in-queue:").append(this.getData(Dict.CURRENT_INDEX)).append(SPLIT);
        }
        if(StringUtils.isNotEmpty(this.messageCode)){
            stringBuffer.append("msg-code:").append(this.getMessageCode()).append(SPLIT);
        }
        if(this.jobId!=null){
            stringBuffer.append("job-id:").append(this.getJobId()).append(SPLIT);
        }
        if (this.getSrcPartyId() != null) {
            stringBuffer.append("src:").append(this.getSrcPartyId()).append(SPLIT);
        }
        if (this.getDesPartyId() != null) {
            stringBuffer.append("des:").append(this.getDesPartyId()).append(SPLIT);
        }
        if (this.getReturnCode() != null) {
            stringBuffer.append("code:").append(this.getReturnCode()).append(SPLIT);
        }
        stringBuffer.append("cost:").append(System.currentTimeMillis() - this.getTimeStamp()).append(SPLIT);
        if (this.getRouterInfo() != null) {
            Protocol protocol = this.getRouterInfo().getProtocol();
            if (protocol != null) {
                if (protocol.equals(Protocol.grpc)) {
                    stringBuffer.append(this.getRouterInfo().getHost() + ":" + this.getRouterInfo().getPort()).append(SPLIT);
                } else if (protocol.equals(Protocol.http)) {
                    stringBuffer.append(this.getRouterInfo().getUrl()).append(SPLIT);
                }
            }
        }
        if (this.getDataSize() != null) {
            stringBuffer.append("size:").append(this.getDataSize()).append(SPLIT);
        }
        if(this.retryTime>1){
            stringBuffer.append("retry:").append(this.retryTime).append(SPLIT);
        }
        if (this.getReturnMsg() != null) {
            stringBuffer.append("msg:").append(this.getReturnMsg());
        }


        return  stringBuffer.toString();
    }
    static final String SPLIT= "|";

}
