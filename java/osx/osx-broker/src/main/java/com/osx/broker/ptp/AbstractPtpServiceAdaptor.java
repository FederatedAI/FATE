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
package com.osx.broker.ptp;


import com.osx.core.context.FateContext;
import com.osx.core.exceptions.ExceptionInfo;
import com.osx.core.service.AbstractServiceAdaptor;
import org.ppc.ptp.Osx;

public abstract class AbstractPtpServiceAdaptor extends AbstractServiceAdaptor<FateContext,Osx.Inbound, Osx.Outbound> {

    @Override
    protected Osx.Outbound transformExceptionInfo(FateContext context, ExceptionInfo exceptionInfo) {

        Osx.Outbound.Builder builder = Osx.Outbound.newBuilder();
        builder.setCode(exceptionInfo.getCode());
        builder.setMessage(exceptionInfo.getMessage());
        return builder.build();
    }

}
