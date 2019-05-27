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

package com.webank.ai.fate.core.api.grpc.client.crud;


import com.google.protobuf.ExperimentalApi;
import org.springframework.context.annotation.Scope;
import org.springframework.stereotype.Component;

@Component
@Scope("prototype")
@ExperimentalApi
@Deprecated
public abstract class BaseCrudDelayedResultClient<T> {
/*
    @Autowired
    private GrpcStubFactory grpcStubFactory;
    @Autowired
    private GrpcStreamObserverFactory grpcStreamObserverFactory;
    @Autowired
    private CallMetaModelFactory callMetaModelFactory;
    @Autowired
    private GrpcAsyncClientContext<T> grpcAsyncClientContext;
    @Autowired
    private ApplicationContext applicationContext;

    private BasicMeta.Endpoint endpoint;
    private Class<T> stubClass;
    private Class grpcClass;
    protected T stub;

    public void init(BasicMeta.Endpoint endpoint) {
        this.grpcAsyncClientContext.setEndpoint(endpoint);
        this.stub = grpcAsyncClientContext.getStub();
    }

    public void init(String ip, int port) {
        BasicMeta.Endpoint.Builder builder = BasicMeta.Endpoint.newBuilder();
        BasicMeta.Endpoint endpoint = builder.setIp(ip).setPort(port).build();

        init(endpoint);
    }

    public Object doCrudRequest(Object model,
                                GrpcCalleeStreamingStubMethodInvoker<T, BasicMeta.CallRequest, BasicMeta.CallResponse> crudRequestProcessor
                                ) {
        BasicMeta.CallRequest request = callMetaModelFactory.createCallRequestFromObject(model);

        GrpcStreamingClientTemplate<T, BasicMeta.CallRequest, BasicMeta.CallResponse> clientTemplate
                = new GrpcStreamingClientTemplate<>();

        ParameterizedType genericType = (ParameterizedType) getClass().getGenericSuperclass();
        Type actualGenericArgument = genericType.getActualTypeArguments()[0];

        grpcAsyncClientContext.setStubClass((Class<T>) actualGenericArgument);

        DelayedResult<BasicMeta.CallResponse> delayedResult = clientTemplate.calleeStreamingRpcWithImmediateDelayedResult(
                request,
                grpcAsyncClientContext,
                DelayedResultUnaryCallStreamObserver.class,
                crudRequestProcessor);

        BasicMeta.CallResponse response = null;
        try {
            response = delayedResult.getResultNow(5, TimeUnit.MINUTES);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

        return callMetaModelFactory.extractModelObject(response);
    }

*/

}
