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

public class ErPartitioner extends BaseProto<Meta.Partitioner> {

    Integer t;
    ByteString body;

    public ErPartitioner(int t, ByteString body) {
        this.t = t;
        this.body = body;
    }

    public ErPartitioner(int t) {
        this(t, ByteString.EMPTY);
    }

    public static ErPartitioner parseFromPb(Meta.Partitioner partitioner) {
        int t = partitioner.getType();
        ByteString body = partitioner.getBody();
        return new ErPartitioner(t, body);
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
    Meta.Partitioner toProto() {
        Meta.Partitioner.Builder builder = Meta.Partitioner.newBuilder()
                .setType(this.t)
                .setBody(this.body);
        return builder.build();
    }
}