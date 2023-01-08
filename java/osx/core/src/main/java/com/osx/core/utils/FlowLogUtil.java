package com.osx.core.utils;

import com.osx.core.context.Context;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class FlowLogUtil {
    static Logger logger = LoggerFactory.getLogger("flow");
    static final String SPLIT= "|";
    public static void printFlowLog(Context context) {
        StringBuffer  stringBuffer = new  StringBuffer();
        if(context.getActionType()!=null){
            stringBuffer.append(context.getActionType()).append(SPLIT);
        }
        if(context.getSessionId()!=null){
            stringBuffer.append("session:").append(context.getSessionId()).append(SPLIT);
        }
        if(context.getTopic()!=null){
            stringBuffer.append("topic:").append(context.getTopic()).append(SPLIT);
        }
        if(context.getRequestMsgIndex()!=null){
            stringBuffer.append("req-offset:").append(context.getRequestMsgIndex()).append(SPLIT);
        }
        if(context.getCurrentMsgIndex()!=null){
            stringBuffer.append("offset-in-queue:").append(context.getCurrentMsgIndex()).append(SPLIT);
        }
        if(context.getSrcPartyId()!=null){
            stringBuffer.append("src:").append(context.getSrcPartyId()).append(SPLIT);
        }
        if(context.getDesPartyId()!=null){
            stringBuffer.append("des:").append(context.getDesPartyId()).append(SPLIT);
        }
        if(context.getReturnCode()!=null){
            stringBuffer.append("code:").append(context.getReturnCode()).append(SPLIT);
        }
        stringBuffer.append("cost:").append(System.currentTimeMillis() - context.getTimeStamp()).append(SPLIT);
        if(context.getRouterInfo()!=null){
            stringBuffer.append("router_info:").append(context.getRouterInfo().getHost() + ":" + context.getRouterInfo().getPort()).append(SPLIT);
        }
        if(context.getDataSize()!=null){
            stringBuffer.append("size:").append(context.getDataSize()).append(SPLIT);
        }
        if(context.getReturnMsg()!=null){
            stringBuffer.append("msg:").append(context.getReturnMsg());
        }
        logger.info(stringBuffer.toString());

    }





}
