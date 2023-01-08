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

package com.osx.core.service;

import com.google.common.collect.Lists;
import com.osx.core.context.Context;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

/**
 * @Description TODO
 * @Author
 **/
public class DefaultInterceptorChain<req, resp> implements InterceptorChain<req, resp> {

    Logger logger = LoggerFactory.getLogger(DefaultInterceptorChain.class);

    List<Interceptor<req, resp>> chain = Lists.newArrayList();

    @Override
    public void addInterceptor(Interceptor<req, resp> interceptor) {
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
    public void doPreProcess(Context context, InboundPackage<req> inboundPackage) throws Exception {
        for (Interceptor<req, resp> interceptor : chain) {
            logger.info("====== {}",interceptor);
            interceptor.doPreProcess(context, inboundPackage);

        }
    }

    /**
     * 后处理即使抛出异常，也将执行完所有
     *
     * @param context
     * @param inboundPackage
     * @param outboundPackage
     * @throws Exception
     */
//    @Override
//    public void doPostProcess(Context context, InboundPackage<req> inboundPackage, OutboundPackage<resp> outboundPackage) throws Exception {
//        for (Interceptor<req, resp> interceptor : chain) {
//            try {
//                interceptor.doPostProcess(context, inboundPackage, outboundPackage);
//            } catch (Throwable e) {
//                logger.error("doPostProcess error", e);
//            }
//        }
//    }
}
