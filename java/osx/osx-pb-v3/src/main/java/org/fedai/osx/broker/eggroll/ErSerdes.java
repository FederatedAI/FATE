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
package org.fedai.osx.broker.eggroll;

import com.google.protobuf.ByteString;
import com.webank.eggroll.core.meta.Meta;

public class ErSerdes extends BaseProto<Meta.Serdes> {

    int t;
    ByteString body;

    public ErSerdes(int t, ByteString body) {
        this.t = t;
        this.body = body;
    }

    public ErSerdes(int t) {
        this(t, ByteString.EMPTY);
    }


    public static ErSerdes parseFromPb(Meta.Serdes serdes) {
        Integer t = serdes.getType();
        ByteString body = serdes.getBody();
        return new ErSerdes(t, body);
    }

    public int getT() {
        return t;
    }

    public void setT(Integer t) {
        this.t = t;
    }

    public ByteString getBody() {
        return body;
    }

    public void setBody(ByteString body) {
        this.body = body;
    }


    @Override
    Meta.Serdes toProto() {
        Meta.Serdes.Builder builder = Meta.Serdes.newBuilder()
                .setType(this.t)
                .setBody(this.body);
        return builder.build();
    }
}