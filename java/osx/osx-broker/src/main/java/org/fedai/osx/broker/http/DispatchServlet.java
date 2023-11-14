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
package org.fedai.osx.broker.http;

import com.google.inject.Inject;
import com.google.inject.Singleton;
import org.apache.commons.lang3.StringUtils;

import org.fedai.osx.broker.util.ContextUtil;
import org.fedai.osx.broker.util.DebugUtil;
import org.fedai.osx.core.constant.PtpHttpHeader;
import org.fedai.osx.core.context.OsxContext;
import org.fedai.osx.core.provider.TechProvider;
import org.fedai.osx.broker.provider.TechProviderRegister;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.servlet.ServletException;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.IOException;
@Singleton
public class DispatchServlet extends HttpServlet {

    Logger logger = LoggerFactory.getLogger(DispatchServlet.class);
    @Inject
    TechProviderRegister  providerRegistry ;

    protected void doGet(HttpServletRequest req, HttpServletResponse resp) throws ServletException, IOException {
        handleInner(req,resp);
    }


    protected void doPost(HttpServletRequest req, HttpServletResponse resp) throws ServletException, IOException {
        handleInner(req,resp);
    }





    private   void  handleInner(HttpServletRequest req, HttpServletResponse  resp) throws IOException {
        //处理get请求
        DebugUtil.printHttpParams(req);
        String protocol = req.getProtocol();
        if (!protocol.endsWith("1.1")) {
            resp.sendError(405, "http.method_get_not_supported");
        }
        OsxContext  osxContext = ContextUtil.buildContextFromHttpRequest(req);
        TechProvider techProvider = providerRegistry.select(osxContext);
            if (techProvider != null) {
                techProvider.processHttpInvoke(osxContext,req, resp);
            } else {
                resp.sendError(404, "tech-provider-code invalid");
            }
        String requestUri = req.getRequestURI();

        switch (requestUri){
            case "/v1/interconn/chan/pop":
                techProvider.processHttpPop(osxContext,req, resp);
                break;
            case "/v1/interconn/chan/push":
                techProvider.processHttpPop(osxContext,req, resp);
                break;
            case "/v1/interconn/chan/peek":
                techProvider.processHttpPeek(osxContext,req, resp);
                break;
            case "/v1/interconn/chan/release":
                techProvider.processHttpRelease(osxContext,req, resp);
                break;
            case "/v1/interconn/chan/invoke":
                techProvider.processHttpInvoke(osxContext,req, resp);
                break;
            default:
                resp.sendError(502, "invalid request "+requestUri);
        }

    }
}
