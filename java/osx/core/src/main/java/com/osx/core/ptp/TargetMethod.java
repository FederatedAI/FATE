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
package com.osx.core.ptp;

public enum TargetMethod {

    //            this.serviceAdaptorConcurrentMap.put("UNARY_CALL",  new UnaryCallService());
//        this.serviceAdaptorConcurrentMap.put("PRODUCE_MSG",new PtpProduceService().addPreProcessor(requestHandleInterceptor));
//        this.serviceAdaptorConcurrentMap.put("ACK_MSG",new PtpAckService().addPreProcessor(requestHandleInterceptor));
//        this.serviceAdaptorConcurrentMap.put("CONSUME_MSG",new PtpConsumeService().addPreProcessor(requestHandleInterceptor));
//        this.serviceAdaptorConcurrentMap.put("QUERY_TOPIC",new PtpQueryTransferQueueService().addPreProcessor(requestHandleInterceptor));
//        this.serviceAdaptorConcurrentMap.put("CANCEL_TOPIC", new PtpCancelTransferService().addPreProcessor(requestHandleInterceptor));
    UNARY_CALL,
    PRODUCE_MSG,
    ACK_MSG,
    CONSUME_MSG,
    QUERY_TOPIC,
    CANCEL_TOPIC,
    PUSH,
    APPLY_TOKEN,
    APPLY_TOPIC




}
