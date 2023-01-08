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
package com.osx.core.provider;

import io.grpc.stub.StreamObserver;
import org.ppc.ptp.Osx;


import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

public interface TechProvider {

    void processHttpInvoke(HttpServletRequest  httpServletRequest,HttpServletResponse httpServletResponse);

    void processGrpcInvoke(Osx.Inbound request,
                           io.grpc.stub.StreamObserver<Osx.Outbound> responseObserver);

    String getProviderId();

    public StreamObserver<Osx.Inbound> processGrpcTransport(Osx.Inbound inbound, io.grpc.stub.StreamObserver<Osx.Outbound> responseObserver);

}
