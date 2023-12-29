package org.fedai.osx.core.service;

import com.google.common.collect.Lists;
import org.fedai.osx.core.context.OsxContext;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

public class DefaultInterceptorChainNew<req, resp> implements InterceptorChainNew<req, resp> {

    Logger logger = LoggerFactory.getLogger(DefaultInterceptorChain.class);

    List<InterceptorNew<req, resp>> chain = Lists.newArrayList();

    @Override
    public void addInterceptor(InterceptorNew<req, resp> interceptor) {
        chain.add(interceptor);
    }

    /**
     * 前处理因为多数是校验逻辑 ， 在这里抛出异常，将中断流程
     *
     * @param context
     * @param inboundPackage
     * @throws Exception
     */
    @Override
    public void doProcess(OsxContext context, req inboundPackage, resp outboundPackage) throws Exception {
        for (InterceptorNew<req, resp> interceptor : chain) {
            if (interceptor != null) {
                interceptor.doProcess(context, inboundPackage, outboundPackage);
            }
        }
    }
}