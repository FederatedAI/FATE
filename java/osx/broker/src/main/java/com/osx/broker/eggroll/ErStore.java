package com.osx.broker.eggroll;


import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.osx.core.utils.JsonUtil;
import com.webank.eggroll.core.meta.Meta;


import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public class ErStore  extends BaseProto<Meta.Store>{

    public String toString(){
        return JsonUtil.object2Json(this);
    }

    public  ErStore(ErStoreLocator  storeLocator,List<ErPartition>  partitions,Map<String,String> options){
        this.storeLocator = storeLocator;
        this.partitions = partitions;
        this.options = options;
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

    ErStoreLocator   storeLocator;
    List<ErPartition> partitions= Lists.newArrayList();
    Map<String,String> options= Maps.newHashMap();


    public ErPartition  getPartition(int  index){
        return partitions.get(index);
    }


    @Override
    Meta.Store toProto() {
        Meta.Store.Builder builder = Meta.Store.newBuilder()
                .setStoreLocator(this.storeLocator.toProto())
                .addAllPartitions(this.partitions.stream().map(ErPartition::toProto).collect(Collectors.toList()))
                .putAllOptions(this.options);
       return  builder.build();
    }


    public static  ErStore parseFromPb(Meta.Store store){
        ErStoreLocator  erStoreLocator = ErStoreLocator.parseFromPb(store.getStoreLocator());

        List<ErPartition>  erPartitions = store.getPartitionsList().stream().map(ErPartition::parseFromPb).collect(Collectors.toList());
        ErStore  erStore = new  ErStore(erStoreLocator,erPartitions,store.getOptionsMap());
        return erStore;
    }
    
    
    public  static void main(String[] args){

//        String namespace ,String name,String  storeType,
//        int totalPartitions ,String partitioner,
//                String serdes

        ErStoreLocator  erStoreLocator =  new  ErStoreLocator("mynamespace",
                "myname","mypath","mystoreType",1,"xxxx","myserdes");

        List<ErPartition>  partitions = Lists.newArrayList();
        ErPartition  erPartition = new  ErPartition(11,null,null,33);
        partitions.add(erPartition);

        ErStore  erStore =  new ErStore(erStoreLocator,partitions,Maps.newHashMap());

        System.err.println(erStore.toProto());

        System.err.println(ErStore.parseFromPb(erStore.toProto()));

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