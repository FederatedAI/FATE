package com.osx.broker.interceptor;


import com.osx.core.context.Context;
import com.osx.core.service.InboundPackage;
import com.osx.core.service.Interceptor;
import com.osx.core.service.OutboundPackage;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;




public class WhiteListInterceptor implements Interceptor {
    Logger logger = LoggerFactory.getLogger(WhiteListInterceptor.class);
    public void doPreProcess(Context context, InboundPackage inboundPackage, OutboundPackage outboundPackage) throws Exception {

        // logger.info("====================== {}  {}",context.getServiceName(),context.getActionType());
        //  context.setSourceIp();
    }
}
