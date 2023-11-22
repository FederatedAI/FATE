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
package org.fedai.osx.broker.grpc;

import com.google.inject.Inject;
import com.google.inject.Singleton;
import io.grpc.Context;
import io.grpc.stub.StreamObserver;

import lombok.extern.slf4j.Slf4j;
import org.fedai.osx.broker.provider.FateTechProvider;
import org.fedai.osx.broker.util.ContextUtil;
import org.fedai.osx.broker.util.DebugUtil;
import org.fedai.osx.broker.util.TransferUtil;

import org.fedai.osx.core.constant.UriConstants;
import org.fedai.osx.core.context.OsxContext;
import org.fedai.osx.core.exceptions.SysException;
import org.fedai.osx.core.provider.TechProvider;
import org.fedai.osx.broker.provider.TechProviderRegister;
import org.ppc.ptp.Osx;
import org.ppc.ptp.PrivateTransferProtocolGrpc;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Map;

@Singleton
@Slf4j
public class PcpInterService extends PrivateTransferProtocolGrpc.PrivateTransferProtocolImplBase {

    @Inject
    TechProviderRegister techProviderRegister;

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
            OsxContext osxContext = new OsxContext();
            ContextUtil.assableContextFromInbound(osxContext);
            TechProvider techProvider = getTechProvider(osxContext);
            techProvider.processGrpcInvoke(osxContext,request, responseObserver);
    }

    private  TechProvider  getTechProvider(OsxContext  context){
        TechProvider techProvider = techProviderRegister.select(context.getTechProviderCode());
        if (techProvider == null) {
            techProvider = techProviderRegister.select("default");
        }
        return  techProvider;
    }


    public class PcpStreamObserver implements io.grpc.stub.StreamObserver<Osx.Inbound> {

        Logger logger = LoggerFactory.getLogger(PcpStreamObserver.class);
        boolean inited = false;
        TechProvider techProvider;
        OsxContext  osxContext;
        StreamObserver<Osx.Outbound> responseObserver;
        StreamObserver<Osx.Inbound> requestObserver;
        public PcpStreamObserver(StreamObserver<Osx.Outbound> responseObserver) {
            this.responseObserver = responseObserver;
        }

        private void init(Osx.Inbound inbound) {

            Map<String, String> metaDataMap = inbound.getMetadataMap();
            // String version = metaDataMap.get(Pcp.Header.Version.name());
            osxContext = new OsxContext();
            ContextUtil.assableContextFromInbound(osxContext);
            logger.info("PcpStreamObserver init {}",metaDataMap);

            String techProviderCode = metaDataMap.get(Osx.Header.TechProviderCode.name());
            techProvider = techProviderRegister.select(osxContext);
            if (techProvider != null) {
                DebugUtil.printGrpcParams(inbound);
                requestObserver = techProvider.processGrpcTransport(osxContext,inbound, responseObserver);
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



    private TechProvider prepare(OsxContext  osxContext){
        ContextUtil.assableContextFromInbound(osxContext);
        TechProvider techProvider = getTechProvider(osxContext);
        return  techProvider;
    }



}
