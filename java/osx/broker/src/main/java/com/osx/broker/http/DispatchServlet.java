package com.osx.broker.http;

import com.osx.broker.ServiceContainer;
import com.osx.core.constant.PtpHttpHeader;
import com.osx.core.provider.TechProvider;
import com.osx.tech.provider.TechProviderRegister;
import org.apache.commons.lang3.StringUtils;
import org.eclipse.jetty.http.HttpHeader;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.servlet.ServletException;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.IOException;

public class DispatchServlet extends HttpServlet {

    Logger logger = LoggerFactory.getLogger(DispatchServlet.class);
    protected void doGet(HttpServletRequest req, HttpServletResponse resp) throws ServletException, IOException {
        String protocol = req.getProtocol();
        if (!protocol.endsWith("1.1")) {
            resp.sendError(405, "http.method_get_not_supported");
        }
        String  techProviderCode =req.getHeader(PtpHttpHeader.TechProviderCode);
        if(StringUtils.isNotEmpty(techProviderCode)){
            TechProvider techProvider = ServiceContainer.techProviderRegister.select(techProviderCode);
            if(techProvider!=null) {
                techProvider.processHttpInvoke(req, resp);
            }else{
                resp.sendError(404,"tech-provider-code invalid");
            }
        }else{
            resp.sendError(404,"tech-provider-code invalid");
        }
        String  requestUri =req.getRequestURI();
        logger.info("receive request uri  {}",requestUri);
    }


    protected void doPost(HttpServletRequest req, HttpServletResponse resp) throws ServletException, IOException {
        String  requestUri =req.getRequestURI();
        logger.info("receive request uri  {}",requestUri);
        String protocol = req.getProtocol();
        if (!protocol.endsWith("1.1")) {
            resp.sendError(405, "http.method_get_not_supported");
        }
        String  techProviderCode =req.getHeader(PtpHttpHeader.TechProviderCode);
        if(StringUtils.isNotEmpty(techProviderCode)){
            TechProvider techProvider = ServiceContainer.techProviderRegister.select(techProviderCode);
            if(techProvider!=null) {
                techProvider.processHttpInvoke(req, resp);
            }else{
                resp.sendError(404,"tech-provider-code invalid");
            }
        }else{
            resp.sendError(404,"tech-provider-code invalid");
        }





    }



}
