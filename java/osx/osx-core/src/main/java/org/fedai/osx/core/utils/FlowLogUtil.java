package org.fedai.osx.core.utils;


import org.fedai.osx.core.context.OsxContext;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class FlowLogUtil {
    static Logger logger = LoggerFactory.getLogger("flow");

    public static void printFlowLog(OsxContext context) {
        try {
            logger.info(context.toString());
        } catch (Throwable ignore) {
        }

    }

}
