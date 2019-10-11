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

package com.webank.ai.fate.driver.federation.factory;

import com.google.common.base.Preconditions;
import com.webank.ai.eggroll.core.constant.StringConstants;
import com.webank.ai.eggroll.core.io.KeyValueStore;
import com.webank.ai.eggroll.core.io.StoreInfo;
import com.webank.ai.eggroll.core.model.Bytes;
import com.webank.ai.eggroll.framework.storage.service.model.RemoteKeyValueStore;
import com.webank.ai.eggroll.framework.storage.service.model.enums.Stores;
import org.apache.commons.lang3.StringUtils;
import org.springframework.context.annotation.Scope;
import org.springframework.stereotype.Component;

import java.util.Properties;

@Component
public class KeyValueStoreFactory {

    public KeyValueStoreBuilder createKeyValueStoreBuilder() {
        return new KeyValueStoreBuilder();
    }


    @Component
    @Scope("prototype")
    public static class KeyValueStoreBuilder {
        private String dataDir;
        private String namespace;
        private String tableName;
        private Stores storageType;
        private String host;
        private int port;
        private int fragment;

        public KeyValueStoreBuilder() {
        }

        public KeyValueStoreBuilder setDataDir(String dataDir) {
            this.dataDir = dataDir;
            return this;
        }

        public KeyValueStoreBuilder setNamespace(String namespace) {
            this.namespace = namespace;
            return this;
        }

        public KeyValueStoreBuilder setTableName(String tableName) {
            this.tableName = tableName;
            return this;
        }

        public KeyValueStoreBuilder setStorageType(Stores storageType) {
            this.storageType = storageType;
            return this;
        }

        public KeyValueStoreBuilder setHost(String host) {
            this.host = host;
            return this;
        }

        public KeyValueStoreBuilder setPort(int port) {
            this.port = port;
            return this;
        }

        public KeyValueStoreBuilder setFragment(int fragment) {
            this.fragment = fragment;
            return this;
        }

        public KeyValueStore<Bytes, byte[]> build(Class<? extends KeyValueStore<Bytes, byte[]>> keyValueStoreImplementationClass) throws ClassNotFoundException {
            Preconditions.checkArgument(StringUtils.isNotBlank(tableName), "tableName cannot be blank");
            StoreInfo storeInfo = StoreInfo.builder()
                    .type(storageType.name())
                    .nameSpace(namespace)
                    .tableName(tableName)
                    .fragment(fragment)
                    .build();

            KeyValueStore<Bytes, byte[]> result = null;

            if (RemoteKeyValueStore.class.equals(keyValueStoreImplementationClass)) {
                result = new RemoteKeyValueStore(storeInfo);
            } else {
                throw new ClassNotFoundException("no KeyValueStore implementation found for class: "
                        + keyValueStoreImplementationClass.getCanonicalName());
            }

            Properties properties = new Properties();
            properties.setProperty(StringConstants.HOST, host);
            properties.setProperty(StringConstants.PORT, String.valueOf(port));

            result.init(properties);

            return result;
        }
    }
}
