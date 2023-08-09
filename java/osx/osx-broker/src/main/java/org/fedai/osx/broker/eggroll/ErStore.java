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

import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.webank.eggroll.core.meta.Meta;
import org.fedai.osx.core.utils.JsonUtil;

import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public class ErStore extends BaseProto<Meta.Store> {

    ErStoreLocator storeLocator;
    List<ErPartition> partitions = Lists.newArrayList();
    Map<String, String> options = Maps.newHashMap();

    public ErStore(ErStoreLocator storeLocator, List<ErPartition> partitions, Map<String, String> options) {
        this.storeLocator = storeLocator;
        this.partitions = partitions;
        this.options = options;
    }

    public static ErStore parseFromPb(Meta.Store store) {
        ErStoreLocator erStoreLocator = ErStoreLocator.parseFromPb(store.getStoreLocator());

        List<ErPartition> erPartitions = store.getPartitionsList().stream().map(ErPartition::parseFromPb).collect(Collectors.toList());
        ErStore erStore = new ErStore(erStoreLocator, erPartitions, store.getOptionsMap());
        return erStore;
    }

    public static void main(String[] args) {

//        String namespace ,String name,String  storeType,
//        int totalPartitions ,String partitioner,
//                String serdes

        ErStoreLocator erStoreLocator = new ErStoreLocator("mynamespace",
                "myname", "mypath", "mystoreType", 1, "xxxx", "myserdes");

        List<ErPartition> partitions = Lists.newArrayList();
        ErPartition erPartition = new ErPartition(11, null, null, 33);
        partitions.add(erPartition);

        ErStore erStore = new ErStore(erStoreLocator, partitions, Maps.newHashMap());


    }

    public String toString() {
        return JsonUtil.object2Json(this);
    }

    public ErStoreLocator getStoreLocator() {
        return storeLocator;
    }

    public void setStoreLocator(ErStoreLocator storeLocator) {
        this.storeLocator = storeLocator;
    }

    public List<ErPartition> getPartitions() {
        return partitions;
    }

    public void setPartitions(List<ErPartition> partitions) {
        this.partitions = partitions;
    }

    public Map<String, String> getOptions() {
        return options;
    }

    public void setOptions(Map<String, String> options) {
        this.options = options;
    }

    public ErPartition getPartition(int index) {
        return partitions.get(index);
    }

    @Override
    Meta.Store toProto() {
        Meta.Store.Builder builder = Meta.Store.newBuilder()
                .setStoreLocator(this.storeLocator.toProto())
                .addAllPartitions(this.partitions.stream().map(ErPartition::toProto).collect(Collectors.toList()))
                .putAllOptions(this.options);
        return builder.build();
    }
}


//  implicit class ErStoreToPbMessage(src: ErStore) extends PbMessageSerializer {
//    override def toProto[T >: PbMessage](): Meta.Store = {
//        val builder = Meta.Store.newBuilder()
//                .setStoreLocator(src.storeLocator.toProto())
//                .addAllPartitions(src.partitions.toList.map(_.toProto()).asJava)
//                .putAllOptions(src.options)
//
//        builder.build()
//    }
//
//    override def toBytes(baseSerializable: BaseSerializable): Array[Byte] =
//    baseSerializable.asInstanceOf[ErStore].toBytes()
//}


//case class ErStore(storeLocator: ErStoreLocator,
//                   partitions: Array[ErPartition] = Array.empty,
//                   options: java.util.Map[String, String] = new ConcurrentHashMap[String, String]())
//        extends MetaRpcMessage {
//        def toPath(delim: String = StringConstants.SLASH): String = storeLocator.toPath(delim = delim)
//
//        def fork(storeLocator: ErStoreLocator): ErStore = {
//        val finalStoreLocator = if (storeLocator == null) storeLocator.fork() else storeLocator
//
//        ErStore(storeLocator = finalStoreLocator, partitions = partitions.map(p => p.copy(storeLocator = finalStoreLocator)))
//        }
//
//        def fork(postfix: String = StringConstants.EMPTY, delimiter: String = StringConstants.UNDERLINE): ErStore = {
//        fork(storeLocator = storeLocator.fork(postfix = postfix, delimiter = delimiter))
//        }
//        }