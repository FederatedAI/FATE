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
import org.fedai.osx.broker.provider.TechProviderRegister;
import org.fedai.osx.broker.router.DefaultFateRouterServiceImpl;
import org.fedai.osx.broker.util.ContextUtil;
import org.fedai.osx.broker.util.DebugUtil;
import org.fedai.osx.broker.util.TransferUtil;
import org.fedai.osx.core.context.OsxContext;
import org.fedai.osx.core.provider.TechProvider;

import javax.servlet.ServletException;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.IOException;
import java.nio.charset.StandardCharsets;

import static org.fedai.osx.core.constant.UriConstants.*;

@Singleton
@Slf4j
public class DispatchServlet extends HttpServlet {
    @Inject
    TechProviderRegister providerRegistry;

    @Inject
    DefaultFateRouterServiceImpl routerService;

    protected void doGet(HttpServletRequest req, HttpServletResponse resp) throws ServletException, IOException {
        handleInner(req, resp);
    }

    protected void doPost(HttpServletRequest req, HttpServletResponse resp) throws ServletException, IOException {
        handleInner(req, resp);
    }


    private void handleInner(HttpServletRequest req, HttpServletResponse resp) throws IOException {
        log.info("handle inner====={}", req.getRequestURI());
        //处理get请求
        DebugUtil.printHttpParams(req);
        String protocol = req.getProtocol();
        if (!protocol.endsWith("1.1")) {
            resp.sendError(405, "http.method_get_not_supported");
        }
        OsxContext osxContext = ContextUtil.buildContextFromHttpRequest(req);

        String requestUri = req.getRequestURI();

        switch (requestUri) {
            case HTTP_POP:
                TechProvider techProvider = providerRegistry.select(osxContext);
                techProvider.processHttpPop(osxContext, req, resp);
                break;
            case HTTP_PUSH:
                techProvider = providerRegistry.select(osxContext);
                techProvider.processHttpPush(osxContext, req, resp);
                break;
            case HTTP_PEEK:
                techProvider = providerRegistry.select(osxContext);
                techProvider.processHttpPeek(osxContext, req, resp);
                break;
            case HTTP_RELEASE:
                techProvider = providerRegistry.select(osxContext);
                techProvider.processHttpRelease(osxContext, req, resp);
                break;
            case HTTP_INVOKE:
                techProvider = providerRegistry.select(osxContext);
                techProvider.processHttpInvoke(osxContext, req, resp);
                break;

            case HTTP_CHANGE_ROUTER:
                byte[] routerContent = TransferUtil.read(req.getInputStream());
                routerService.saveRouterTable(osxContext, new String(routerContent));
                TransferUtil.writeHttpRespose(resp, "ok", "ok", "ok".getBytes(StandardCharsets.UTF_8));
            default:
                resp.sendError(502, "invalid request " + requestUri);
        }

    }
}
