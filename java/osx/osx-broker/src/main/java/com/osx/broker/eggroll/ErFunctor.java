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
package com.osx.broker.eggroll;

import com.google.protobuf.ByteString;
import com.webank.eggroll.core.meta.Meta;

import java.util.Map;

public class ErFunctor extends BaseProto<Meta.Functor> {

    String name;
    String serdes;
    byte[] body;
    Map<String, String> options;
    public ErFunctor(String name, String serdes, byte[] body, Map<String, String> options) {
        this.name = name;
        this.serdes = serdes;
        this.body = body;
        this.options = options;

    }

    public static ErFunctor parseFromPb(Meta.Functor functor) {
        if (functor == null)
            return null;
        String name = functor.getName();
        ByteString bodyByteString = functor.getBody();
        Map<String, String> options = functor.getOptionsMap();
        String serdes = functor.getSerdes();
        ErFunctor erFunctor = new ErFunctor(name, serdes, bodyByteString != null ? bodyByteString.toByteArray() : null, options);
        return erFunctor;
    }

    @Override
    Meta.Functor toProto() {
        return Meta.Functor.newBuilder().
                setName(this.name).
                setSerdes(this.serdes).putAllOptions(options).
                setBody(ByteString.copyFrom(body)).build();
    }
}

//case class ErFunctor(name: String = StringConstants.EMPTY,
//                     serdes: String = StringConstants.EMPTY,
//                     body: Array[Byte]) extends RpcMessage
