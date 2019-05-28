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

package com.webank.ai.eggroll.core.constant;

import com.google.protobuf.ByteString;
import com.google.protobuf.Descriptors;
import com.webank.ai.eggroll.api.computing.processor.Processor;
import com.webank.ai.eggroll.core.io.StoreInfo;
import io.grpc.Context;
import io.grpc.Metadata;

import java.util.Map;

public class MetaConstants {

    public static final CompositeHeaderKey STORE_TYPE = CompositeHeaderKey.from("STORE_TYPE");
    public static final CompositeHeaderKey TABLE_NAME = CompositeHeaderKey.from("TABLE_NAME");
    public static final CompositeHeaderKey NAME_SPACE = CompositeHeaderKey.from("NAME_SPACE");
    public static final CompositeHeaderKey FRAGMENT = CompositeHeaderKey.from("FRAGMENT");

    private static final CompositeHeaderKey[] STORE_META = {
            STORE_TYPE,
            TABLE_NAME,
            NAME_SPACE,
            FRAGMENT
    };


    public static Context updateContext(Metadata metadata, Context context) {
        Context rtn = context;
        for (CompositeHeaderKey compositeHeaderKey : STORE_META) {
            rtn = rtn.withValue(compositeHeaderKey.asContextKey(), metadata.get(compositeHeaderKey.asMetaKey()));
        }
        return rtn;
    }


    public static void updateMeta(Processor.TaskInfo processInfo, Metadata metadata) {
        for (Map.Entry<Descriptors.FieldDescriptor, Object> entry : processInfo.getAllFields().entrySet()) {
            CompositeHeaderKey key = CompositeHeaderKey.from(entry.getKey().getName());
            if (entry.getValue() instanceof ByteString) {
                metadata.put(key.asMetaBytesKey(), ((ByteString) entry.getValue()).toByteArray());
            } else {
                metadata.put(key.asMetaKey(), entry.getValue().toString());
            }
        }
    }

    public static Metadata createMetadataFromStoreInfo(StoreInfo storeInfo) {
        Metadata result = null;
        if (storeInfo != null) {
            result = new Metadata();
            result.put(STORE_TYPE.asMetaKey(), storeInfo.getType());
            result.put(NAME_SPACE.asMetaKey(), storeInfo.getNameSpace());
            result.put(TABLE_NAME.asMetaKey(), storeInfo.getTableName());
            if (storeInfo.getFragment() != null) {
                result.put(FRAGMENT.asMetaKey(), storeInfo.getFragment().toString());
            }
        }
        return result;
    }


    public static class CompositeHeaderKey {

        private final String keyString;
        private final Metadata.Key<String> stringMetaKey;
        private final Metadata.Key<byte[]> bytesMetaKey;
        private final Context.Key<String> stringContextKey;

        private CompositeHeaderKey(String key) {
            this.keyString = key;
            this.stringMetaKey = Metadata.Key.<String>of(keyString, Metadata.ASCII_STRING_MARSHALLER);
            this.bytesMetaKey = Metadata.Key.<byte[]>of(keyString + Metadata.BINARY_HEADER_SUFFIX, Metadata.BINARY_BYTE_MARSHALLER);
            this.stringContextKey = Context.<String>key(keyString);

        }

        public static CompositeHeaderKey from(String key) {
            return new CompositeHeaderKey(key);
        }


        public Metadata.Key<String> asMetaKey() {
            return stringMetaKey;
        }

        public Metadata.Key<byte[]> asMetaBytesKey() {
            return bytesMetaKey;
        }

        public Context.Key<String> asContextKey() {
            return stringContextKey;
        }


        public String asString() {
            return keyString;
        }

        @Override
        public String toString() {
            return keyString;
        }

    }


}
