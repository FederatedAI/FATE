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

public class ErJobIO extends BaseProto<Meta.JobIO> {

    ErStore erStore;
    ErSerdes keySerdes;
    ErSerdes valueSerdes;
    ErPartitioner partitioner;

    public ErJobIO(ErStore erStore) {
        this.erStore = erStore;
        this.keySerdes = new ErSerdes(0);
        this.valueSerdes = new ErSerdes(0);
        this.partitioner = new ErPartitioner(0);
    }


    public ErJobIO(ErStore erStore, ErSerdes keySerdes, ErSerdes valueSerdes, ErPartitioner partitioner) {
        this.erStore = erStore;
        this.keySerdes = keySerdes;
        this.valueSerdes = valueSerdes;
        this.partitioner = partitioner;
    }

    public static ErJobIO parseFromPb(Meta.JobIO jobIO) {
        ErStore erStore = ErStore.parseFromPb(jobIO.getStore());
        ErSerdes key_serdes = ErSerdes.parseFromPb(jobIO.getKeySerdes());
        ErSerdes value_serdes = ErSerdes.parseFromPb(jobIO.getValueSerdes());
        ErPartitioner partitioner = ErPartitioner.parseFromPb(jobIO.getPartitioner());
        ErJobIO erJobIO = new ErJobIO(erStore, key_serdes, value_serdes, partitioner);
        return erJobIO;
    }

    public ErSerdes getKeySerdes() {
        return keySerdes;
    }

    public void setKeySerdes(ErSerdes keySerdes) {
        this.keySerdes = keySerdes;
    }

    public ErSerdes getValueSerdes() {
        return valueSerdes;
    }

    public void setValueSerdes(ErSerdes valueSerdes) {
        this.valueSerdes = valueSerdes;
    }

    public ErPartitioner getPartitioner() {
        return partitioner;
    }

    public void setPartitioner(ErPartitioner partitioner) {
        this.partitioner = partitioner;
    }

    public ErStore getErStore() {
        return erStore;
    }

    public void setErStore(ErStore erStore) {
        this.erStore = erStore;
    }

    @Override
    Meta.JobIO toProto() {
        Meta.JobIO.Builder builder = Meta.JobIO.newBuilder();
        builder.setStore(erStore.toProto())
                .setKeySerdes(keySerdes.toProto())
                .setValueSerdes(valueSerdes.toProto())
                .setPartitioner(partitioner.toProto());
        return builder.build();
    }
}
