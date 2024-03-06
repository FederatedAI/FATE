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

import com.webank.eggroll.core.meta.Meta;
import org.apache.commons.lang3.StringUtils;

public class ErStoreLocator extends BaseProto<Meta.StoreLocator> {

    String storeType;
    String namespace;
    String name;
    String path;
    int totalPartitions;

    int KeySerdesType;
    int ValueSerdesType;
    int PartitionerType;

    public ErStoreLocator(String namespace, String name,
                          String path,
                          String storeType,
                          int totalPartitions,
                          String partitioner,
                          String serdes
    ) {
        this.namespace = namespace;
        this.name = name;
        this.path = path;
        this.storeType = storeType;
        this.totalPartitions = totalPartitions;
        this.KeySerdesType = 0;
        this.ValueSerdesType = 0;
        this.PartitionerType = 0;

    }

    public ErStoreLocator(String namespace, String name,
                          String path,
                          String storeType,
                          int totalPartitions,
                          int KeySerdesType,
                          int ValueSerdesType,
                          int PartitionerType
    ) {
        this.storeType = storeType;
        this.namespace = namespace;
        this.name = name;
        this.path = path;
        this.totalPartitions = totalPartitions;
        this.KeySerdesType = KeySerdesType;
        this.ValueSerdesType = ValueSerdesType;
        this.PartitionerType = PartitionerType;
    }

    public static ErStoreLocator parseFromPb(Meta.StoreLocator storeLocator) {


        ErStoreLocator erStoreLocator = new ErStoreLocator(
                storeLocator.getNamespace(),
                storeLocator.getName(),
                storeLocator.getPath(),
                storeLocator.getStoreType(),
                storeLocator.getTotalPartitions(),
                storeLocator.getKeySerdesType(),
                storeLocator.getValueSerdesType(),
                storeLocator.getPartitionerType()
        );
        return erStoreLocator;

    }

    public String getStoreType() {
        return storeType;
    }

    public void setStoreType(String storeType) {
        this.storeType = storeType;
    }

    public String getNamespace() {
        return namespace;
    }

    public void setNamespace(String namespace) {
        this.namespace = namespace;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getPath() {
        return path;
    }

    public void setPath(String path) {
        this.path = path;
    }

    public int getTotalPartitions() {
        return totalPartitions;
    }

    public void setTotalPartitions(int totalPartitions) {
        this.totalPartitions = totalPartitions;
    }

    public int getKeySerdesType() {
        return KeySerdesType;
    }

    public void setKeySerdesType(int keySerdesType) {
        KeySerdesType = keySerdesType;
    }

    public int getValueSerdesType() {
        return ValueSerdesType;
    }

    public void setValueSerdesType(int valueSerdesType) {
        ValueSerdesType = valueSerdesType;
    }

    public int getPartitionerType() {
        return PartitionerType;
    }

    public void setPartitionerType(int partitionerType) {
        PartitionerType = partitionerType;
    }

    String toPath(String delim) {
        if (!StringUtils.isBlank(path)) {
            return path;
        } else {
            return String.join(delim, storeType, namespace, name);
        }
    }

    @Override
    Meta.StoreLocator toProto() {

        Meta.StoreLocator.Builder builder = Meta.StoreLocator.newBuilder();

        return builder.setName(name)
                .setNamespace(namespace)
                .setStoreType(storeType)
                .setPath(path)
                .setTotalPartitions(totalPartitions)
                .setKeySerdesType(KeySerdesType)
                .setValueSerdesType(ValueSerdesType)
                .setPartitionerType(PartitionerType).
                        build();
    }

    public String toString() {
        return PbUtils.object2Json(this);
    }


}



