package org.fedai.osx.broker.util;


import org.fedai.osx.core.config.MetaInfo;
import org.fedai.osx.core.context.Protocol;
import org.fedai.osx.core.utils.JsonUtil;
import org.ppc.ptp.Osx;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.servlet.http.HttpServletRequest;
import java.io.BufferedReader;
import java.io.IOException;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.Map;

public class DebugUtil {

    static Logger logger = LoggerFactory.getLogger(DebugUtil.class);

    public static void printGrpcParams(Osx.Inbound request) {
        try {
            if (MetaInfo.PROTOCOL_PARAMS_PRINT) {
                logger.info("【{}】====> {}", Protocol.grpc.name(), JsonUtil.object2Json(request.getMetadataMap()));
            }
        } catch (Exception e) {
            logger.error("DebugUtil.printGrpcParams error : ", e);
        }
    }

    public static void printHttpParams(HttpServletRequest request) {
        try {
            if (MetaInfo.PROTOCOL_PARAMS_PRINT) {
                StringBuilder info = new StringBuilder("【" + Protocol.http.name() + "】====> " + "(uri) = " + request.getRequestURI() + "\n(head) = ");
                Enumeration<String> headerNames = request.getHeaderNames();
                Map<String, Object> headMap = new HashMap<>();
                if (headerNames.hasMoreElements()) {
                    String headerName = headerNames.nextElement();
                    headMap.put(headerName, request.getHeader(headerName));
                }
                info.append("\n").append(JsonUtil.object2Json(headMap)).append("\n(body) = ");
                try (BufferedReader reader = request.getReader()) {
                    String line;
                    while ((line = reader.readLine()) != null) {
                        info.append(line);
                    }
                } catch (IOException e) {
                    e.printStackTrace();
                }
                logger.info(info.toString());
            }
        } catch (Exception e) {
            logger.error("DebugUtil.printGrpcParams error : ", e);
        }
    }
}
