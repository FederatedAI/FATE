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
        if(context.getCaseId()!=null){
            stringBuffer.append("seq:").append(context.getCaseId()).append(SPLIT);
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

//        logger.info("{}|{}|seq:{}|topic:{}|src:{}|des:{}|" +
//                        "code:{}|cost:{}|send_to:{}|" +
//                        "size:{}|msg:{}",
//                context.getActionType(), context.getSessionId(), context.getCaseId(), context.getTopic(), context.getSrcPartyId(),
//                context.getDesPartyId(), context.getReturnCode(), (System.currentTimeMillis() - context.getTimeStamp())
//                , context.getRouterInfo() != null ? context.getRouterInfo().getHost() + ":" + context.getRouterInfo().getPort() : ""
//                , context.getDataSize(), context.getReturnMsg()!=null?  context.getReturnMsg():""
//        );
    }


//    public static  void  printFlowLogForClusterManager(Context  context){
//        logger.info("{}|{}|{}|{}|{}|{}",
//                context.getActionType(), context.getInstanceId(),context.getTransferId()
//                ,context.getReturnCode(),System.currentTimeMillis()-context.getTimeStamp(),
//                context.getResultData()
//        );
//    }

    public static void printFlowLogForConsumer(Context context) {
        logger.info("{}|{}|{}|{}|" +
                        "{}|{}|{}|" +
                        "{}|{}",
                context.getActionType(), context.getTopic(), context.getRequestMsgIndex(),
                context.getCurrentMsgIndex(), context.getReturnCode(), System.currentTimeMillis() - context.getTimeStamp()
                , context.getRouterInfo() != null ? context.getRouterInfo().getHost() + ":" + context.getRouterInfo().getPort() : ""
                , context.getDataSize(), context.getException() != null ? context.getException().getMessage() : "", ""
        );
    }


    public static void printFlowLogForAck(Context context) {

        logger.info("{}|{}|{}|{}|" +
                        "{}|{}|{}|" +
                        "{}|{}",
                context.getActionType(), context.getTopic(), context.getRequestMsgIndex(),
                context.getCurrentMsgIndex(), context.getReturnCode(), System.currentTimeMillis() - context.getTimeStamp()
                , context.getRouterInfo() != null ? context.getRouterInfo().getHost() + ":" + context.getRouterInfo().getPort() : ""
                , context.getDataSize(), context.getException() != null ? context.getException().getMessage() : "", ""
        );
    }


}
