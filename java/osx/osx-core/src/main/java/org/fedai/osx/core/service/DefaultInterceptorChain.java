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

package org.fedai.osx.core.service;

import com.google.common.collect.Lists;
import org.fedai.osx.api.context.Context;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

/**
 * @Description TODO
 * @Author
 **/
public class DefaultInterceptorChain<ctx extends Context, req, resp> implements InterceptorChain<ctx, req, resp> {

    Logger logger = LoggerFactory.getLogger(DefaultInterceptorChain.class);

    List<Interceptor<ctx, req, resp>> chain = Lists.newArrayList();

    @Override
    public void addInterceptor(Interceptor<ctx, req, resp> interceptor) {
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
    public void doProcess(ctx context, InboundPackage<req> inboundPackage,OutboundPackage<resp> outboundPackage) throws Exception {
        for (Interceptor<ctx, req, resp> interceptor : chain) {
            if (interceptor != null) {
                interceptor.doProcess(context, inboundPackage,outboundPackage);
            }
        }
    }
}
