package com.osx.core.utils;
import com.osx.core.context.Context;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class FlowLogUtil {
    static Logger logger = LoggerFactory.getLogger("flow");

    public static  void printFlowLog(Context context) {

        logger.info("{}|{}|seq:{}|topic:{}|src:{}|des:{}|" +
                        "code:{}|cost:{}|send_to:{}|" +
                        "size:{}|msg:{}",
                context.getActionType(),context.getSessionId(),context.getCaseId(), context.getTopic(), context.getSrcPartyId(),
                context.getDesPartyId(),context.getReturnCode(),(System.currentTimeMillis()-context.getTimeStamp())
                , context.getRouterInfo() != null ? context.getRouterInfo().getHost()+":"+context.getRouterInfo().getPort() : ""
                ,context.getDataSize() ,context.getException()!=null?context.getException().getMessage():"",""
                );
    }


//    public static  void  printFlowLogForClusterManager(Context  context){
//        logger.info("{}|{}|{}|{}|{}|{}",
//                context.getActionType(), context.getInstanceId(),context.getTransferId()
//                ,context.getReturnCode(),System.currentTimeMillis()-context.getTimeStamp(),
//                context.getResultData()
//        );
//    }

    public static  void printFlowLogForConsumer(Context context) {
        logger.info("{}|{}|{}|{}|" +
                        "{}|{}|{}|" +
                        "{}|{}",
                context.getActionType(), context.getTopic(), context.getRequestMsgIndex(),
                context.getCurrentMsgIndex(),context.getReturnCode(),System.currentTimeMillis()-context.getTimeStamp()
                , context.getRouterInfo() != null ? context.getRouterInfo().getHost()+":"+context.getRouterInfo().getPort() : ""
                ,context.getDataSize() ,context.getException()!=null?context.getException().getMessage():"",""
        );
    }


    public static  void printFlowLogForAck(Context context) {

        logger.info("{}|{}|{}|{}|" +
                        "{}|{}|{}|" +
                        "{}|{}",
                context.getActionType(), context.getTopic(), context.getRequestMsgIndex(),
                context.getCurrentMsgIndex(),context.getReturnCode(),System.currentTimeMillis()-context.getTimeStamp()
                , context.getRouterInfo() != null ? context.getRouterInfo().getHost()+":"+context.getRouterInfo().getPort() : ""
                ,context.getDataSize() ,context.getException()!=null?context.getException().getMessage():"",""
        );
    }




}
