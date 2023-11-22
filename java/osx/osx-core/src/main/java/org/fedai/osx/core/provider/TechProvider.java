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
package org.fedai.osx.core.provider;

import io.grpc.stub.StreamObserver;
import org.fedai.osx.core.context.OsxContext;
import org.ppc.ptp.Osx;

import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

public interface TechProvider {
    //用于处理http1.X请求
    default  void processHttpInvoke(OsxContext context, HttpServletRequest  httpServletRequest, HttpServletResponse httpServletResponse){ };
    //用于处理grpc非流式请求
    void processGrpcInvoke(OsxContext context,Osx.Inbound request,
                           io.grpc.stub.StreamObserver<Osx.Outbound> responseObserver);

    default void processHttpPeek(OsxContext context, HttpServletRequest  httpServletRequest, HttpServletResponse httpServletResponse){ };
    default void processHttpPush(OsxContext context, HttpServletRequest  httpServletRequest, HttpServletResponse httpServletResponse){ };
    default void processHttpPop(OsxContext context, HttpServletRequest  httpServletRequest, HttpServletResponse httpServletResponse){ };
    default void processHttpRelease(OsxContext context, HttpServletRequest  httpServletRequest, HttpServletResponse httpServletResponse){ };
    //用于处理grpc流式请求
   default public StreamObserver<Osx.Inbound> processGrpcTransport(OsxContext context,Osx.Inbound inbound, StreamObserver<Osx.Outbound> responseObserver){return null;};

   void processGrpcPeek(OsxContext context,Osx.PeekInbound inbound, io.grpc.stub.StreamObserver<Osx.TransportOutbound> responseObserver);

    void processGrpcPush(OsxContext context,Osx.PushInbound inbound, io.grpc.stub.StreamObserver<Osx.TransportOutbound> responseObserver);

    void processGrpcPop(OsxContext context,Osx.PopInbound inbound, io.grpc.stub.StreamObserver<Osx.TransportOutbound> responseObserver);

    void processGrpcRelease(OsxContext  context ,Osx.ReleaseInbound inbound, io.grpc.stub.StreamObserver<Osx.TransportOutbound> responseObserver);





}
