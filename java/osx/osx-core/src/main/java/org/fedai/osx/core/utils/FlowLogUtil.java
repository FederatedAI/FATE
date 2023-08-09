package org.fedai.osx.core.utils;

import org.fedai.osx.api.context.Context;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class FlowLogUtil {
    static Logger logger = LoggerFactory.getLogger("flow");

    public static void printFlowLog(Context context) {
        try {
            logger.info(context.toString());
        }catch (Throwable ignore){
        }

    }





}
