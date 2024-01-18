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

public class ErPartition extends BaseProto<Meta.Partition> {

    Integer id;
    ErStoreLocator storeLocator;
    ErProcessor processor;
    int rankInNode;

    public ErPartition(Integer id, ErStoreLocator storeLocator, ErProcessor processor, int rankInNode) {
        this.id = id;
        this.storeLocator = storeLocator;
        this.processor = processor;
        this.rankInNode = rankInNode;

    }

    public static ErPartition parseFromPb(Meta.Partition partition) {

        if (partition != null) {
            Meta.StoreLocator storeLocator = partition.getStoreLocator();
            ErStoreLocator erStoreLocator = ErStoreLocator.parseFromPb(storeLocator);
            ErProcessor erProcessor = ErProcessor.parseFromPb(partition.getProcessor());
            ErPartition erPartition = new ErPartition(partition.getId(), erStoreLocator, erProcessor, partition.getRankInNode());
            return erPartition;
        } else {
            return null;
        }
    }

    public Integer getId() {
        return id;
    }

    public void setId(Integer id) {
        this.id = id;
    }

    public ErStoreLocator getStoreLocator() {
        return storeLocator;
    }

    public void setStoreLocator(ErStoreLocator storeLocator) {
        this.storeLocator = storeLocator;
    }

    public ErProcessor getProcessor() {
        return processor;
    }

    public void setProcessor(ErProcessor processor) {
        this.processor = processor;
    }

    public int getRankInNode() {
        return rankInNode;
    }

    public void setRankInNode(int rankInNode) {
        this.rankInNode = rankInNode;
    }

    public String toPath(String delim) {
        return String.join(delim, storeLocator.toPath(delim = delim), id.toString());
    }

    @Override
    Meta.Partition toProto() {

        Meta.Partition.Builder builder = Meta.Partition.newBuilder();
        builder.setId(id);
        builder.setRankInNode(rankInNode);
        if (storeLocator != null)
            builder.setStoreLocator(storeLocator.toProto());
        if (processor != null) {
            builder.setProcessor(this.processor.toProto());
        }
        return builder.build();
    }
}


//case class ErPartition(id: Int,
//                       storeLocator: ErStoreLocator = null,
//                       processor: ErProcessor = null,
//                       rankInNode: Int = -1) extends MetaRpcMessage {
//    def toPath(delim: String = StringConstants.SLASH): String = String.join(delim, storeLocator.toPath(delim = delim), id.toString)
//}
