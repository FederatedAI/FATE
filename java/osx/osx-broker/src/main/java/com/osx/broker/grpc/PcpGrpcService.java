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
package com.osx.broker.grpc;

import com.osx.broker.ServiceContainer;
import com.osx.broker.util.DebugUtil;
import com.osx.core.exceptions.SysException;
import com.osx.core.provider.TechProvider;
import io.grpc.stub.StreamObserver;
import org.ppc.ptp.Osx;
import org.ppc.ptp.PrivateTransferProtocolGrpc;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Map;
public class PcpGrpcService extends PrivateTransferProtocolGrpc.PrivateTransferProtocolImplBase {

    /**
     * 流式接口
     *
     * @param responseObserver
     * @return
     */
    public io.grpc.stub.StreamObserver<Osx.Inbound> transport(
            io.grpc.stub.StreamObserver<Osx.Outbound> responseObserver) {
        return new PcpStreamObserver(responseObserver);
    }

    /**
     * 同步接口
     *
     * @param request
     * @param responseObserver
     */
    public void invoke(Osx.Inbound request,
                       io.grpc.stub.StreamObserver<Osx.Outbound> responseObserver) {

        DebugUtil.printGrpcParams(request);
        Map<String, String> metaDataMap = request.getMetadataMap();
        String techProviderCode = metaDataMap.get(Osx.Header.TechProviderCode.name());
        TechProvider techProvider = ServiceContainer.techProviderRegister.select(techProviderCode);
        if (techProvider != null) {
            techProvider.processGrpcInvoke(request, responseObserver);
        }
    }

    public class PcpStreamObserver implements io.grpc.stub.StreamObserver<Osx.Inbound> {

        Logger logger = LoggerFactory.getLogger(PcpStreamObserver.class);
        boolean inited = false;
        TechProvider techProvider;
        StreamObserver<Osx.Outbound> responseObserver;
        StreamObserver<Osx.Inbound> requestObserver;
        public PcpStreamObserver(StreamObserver<Osx.Outbound> responseObserver) {
            this.responseObserver = responseObserver;
        }

        private void init(Osx.Inbound inbound) {

            Map<String, String> metaDataMap = inbound.getMetadataMap();
            // String version = metaDataMap.get(Pcp.Header.Version.name());

            logger.info("PcpStreamObserver init {}",metaDataMap);

            String techProviderCode = metaDataMap.get(Osx.Header.TechProviderCode.name());
            techProvider = ServiceContainer.techProviderRegister.select(techProviderCode);
            if (techProvider != null) {
                DebugUtil.printGrpcParams(inbound);
                requestObserver = techProvider.processGrpcTransport(inbound, responseObserver);
            } else {
                //抛出异常
                logger.error("can not found TechProvider of {}",techProviderCode);
                throw  new SysException("invalid TechProviderCode "+techProviderCode);
            }
            inited = true;
            logger.info("PcpStreamObserver init over");
        }

        @Override
        public void onNext(Osx.Inbound inbound) {
            if (!inited) {
                init(inbound);
            }
            if (requestObserver != null) {
                requestObserver.onNext(inbound);
            } else {
                throw new SysException("requestObserver is null");
            }
        }

        @Override
        public void onError(Throwable throwable) {
            if (requestObserver != null) {
                requestObserver.onError(throwable);
            }
        }

        @Override
        public void onCompleted() {
            if (requestObserver != null) {
                requestObserver.onCompleted();
            }
        }
    }


}
