package org.fedai.osx.broker.http;

import com.google.inject.Inject;
import com.google.inject.Singleton;
import lombok.Data;
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

@Data
@Singleton
public class InterServlet extends InnerServlet {

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
                case HTTP_INVOKE:
                    techProvider.processHttpInvoke(osxContext, req, resp,true);
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

