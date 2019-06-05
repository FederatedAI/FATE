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

package com.webank.ai.eggroll.framework.roll.service.async.processor;

import com.webank.ai.eggroll.api.core.BasicMeta;
import com.webank.ai.eggroll.api.computing.processor.Processor;
import com.webank.ai.eggroll.api.storage.StorageBasic;
import com.webank.ai.eggroll.framework.roll.api.grpc.client.EggProcessServiceClient;
import org.springframework.context.annotation.Scope;
import org.springframework.stereotype.Component;

@Component
@Scope("prototype")
public class GlomServiceProcessor extends BaseProcessServiceProcessor<Processor.UnaryProcess, StorageBasic.StorageLocator> {
    public GlomServiceProcessor(EggProcessServiceClient eggProcessServiceClient, Processor.UnaryProcess request, BasicMeta.Endpoint processorEndpoint) {
        super(eggProcessServiceClient, request, processorEndpoint);
    }

    @Override
    public StorageBasic.StorageLocator call() throws Exception {
        return eggProcessServiceClient.glom(request, processorEndpoint);
    }
}
