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
package com.osx.broker.http;

import com.osx.broker.ServiceContainer;
import com.osx.core.constant.PtpHttpHeader;
import com.osx.core.provider.TechProvider;
import org.apache.commons.lang3.StringUtils;
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
        //处理get请求
        String protocol = req.getProtocol();
        if (!protocol.endsWith("1.1")) {
            resp.sendError(405, "http.method_get_not_supported");
        }
        String techProviderCode = req.getHeader(PtpHttpHeader.TechProviderCode);
        if (StringUtils.isNotEmpty(techProviderCode)) {
            TechProvider techProvider = ServiceContainer.techProviderRegister.select(techProviderCode);
            if (techProvider != null) {
                techProvider.processHttpInvoke(req, resp);
            } else {
                resp.sendError(404, "tech-provider-code invalid");
            }
        } else {
            resp.sendError(404, "tech-provider-code invalid");
        }
        String requestUri = req.getRequestURI();
        logger.info("receive request uri  {}", requestUri);
    }


    protected void doPost(HttpServletRequest req, HttpServletResponse resp) throws ServletException, IOException {
        //处理post请求
        String requestUri = req.getRequestURI();
        //logger.info("receive request uri  {}",requestUri);
        String protocol = req.getProtocol();
        if (!protocol.endsWith("1.1")) {
            resp.sendError(405, "http.method_get_not_supported");
        }
        String techProviderCode = req.getHeader(PtpHttpHeader.TechProviderCode);
        if (StringUtils.isNotEmpty(techProviderCode)) {
            TechProvider techProvider = ServiceContainer.techProviderRegister.select(techProviderCode);
            if (techProvider != null) {
                techProvider.processHttpInvoke(req, resp);
            } else {
                resp.sendError(404, "tech-provider-code invalid");
            }
        } else {
            resp.sendError(404, "tech-provider-code invalid");
        }
    }


}
