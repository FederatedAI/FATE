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
import lombok.extern.slf4j.Slf4j;
import org.fedai.osx.broker.provider.FateTechProvider;
import org.fedai.osx.broker.provider.TechProviderRegister;

import org.fedai.osx.broker.util.ContextUtil;
import org.fedai.osx.broker.util.DebugUtil;
import org.fedai.osx.broker.util.TransferUtil;
import org.fedai.osx.core.context.OsxContext;
import org.fedai.osx.core.exceptions.ErrorMessageUtil;
import org.fedai.osx.core.exceptions.ExceptionInfo;
import org.fedai.osx.core.provider.TechProvider;
import org.fedai.osx.core.utils.JsonUtil;

import javax.servlet.ServletException;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.IOException;
import java.nio.charset.StandardCharsets;

import static org.fedai.osx.core.constant.UriConstants.*;

@Singleton
@Slf4j
public class InnerServlet extends HttpServlet {
    @Inject
    TechProviderRegister providerRegistry;

    protected void doGet(HttpServletRequest req, HttpServletResponse resp) throws ServletException, IOException {
        dispatch(req, resp);
    }

    protected void doPost(HttpServletRequest req, HttpServletResponse resp) throws ServletException, IOException {
        dispatch(req, resp);
    }

    protected void dispatch(HttpServletRequest req, HttpServletResponse resp) throws IOException {
        OsxContext osxContext = ContextUtil.buildContextFromHttpRequest(req);
        try {
            DebugUtil.printHttpParams(req);
            String protocol = req.getProtocol();
            if (!protocol.endsWith("1.1")) {
                resp.sendError(405, "http.method_get_not_supported");
            }
            String requestUri = req.getRequestURI();
            TechProvider techProvider = providerRegistry.select(osxContext);
            switch (requestUri) {
                case HTTP_POP:
                    techProvider.processHttpPop(osxContext, req, resp);
                    break;
                case HTTP_PUSH:
                    techProvider.processHttpPush(osxContext, req, resp);
                    break;
                case HTTP_PEEK:
                    techProvider.processHttpPeek(osxContext, req, resp);
                    break;
                case HTTP_RELEASE:
                    techProvider.processHttpRelease(osxContext, req, resp);
                    break;
                case HTTP_INVOKE:
                    techProvider.processHttpInvoke(osxContext, req, resp);
                    break;

                case  HTTP_SET_ROUTER:
                case  HTTP_GET_ROUTER:
                case  HTTP_ADD_ROUTER:
                    osxContext.setUri(requestUri);
                    ((FateTechProvider) techProvider).processRouterOperation(osxContext, req, resp);
                    break;
                default:
                    resp.sendError(502, "invalid request " + requestUri);
            }
        }catch (Exception e){
            ExceptionInfo exceptionInfo = ErrorMessageUtil.handleExceptionExceptionInfo(osxContext,e);
            TransferUtil.writeHttpRespose(resp, exceptionInfo.getCode(), exceptionInfo.getMessage(), JsonUtil.object2Json(exceptionInfo).getBytes(StandardCharsets.UTF_8));
        }

    }
}
