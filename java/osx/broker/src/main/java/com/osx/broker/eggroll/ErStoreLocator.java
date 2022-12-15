package com.osx.broker.eggroll;


import com.osx.core.utils.JsonUtil;
import com.webank.eggroll.core.meta.Meta;
import org.apache.commons.lang3.StringUtils;

public class ErStoreLocator  extends  BaseProto<Meta.StoreLocator>{


    public ErStoreLocator(String namespace ,String name,
                          String path,
                          String  storeType,
                          int totalPartitions ,String partitioner,
                            String serdes
    ){
        this.namespace = namespace;
        this.name = name;
        this.path = path;
        this.storeType = storeType;
        this.partitioner = partitioner;
        this.totalPartitions = totalPartitions;
        this.serdes = serdes;

    }


//    (
//    namespace = namespace,
//    name = name,
//    storeType = storeType,
//    totalPartitions = totalPartitions,
//    partitioner = options.getOrElse(StringConstants.PARTITIONER, PartitionerTypes.BYTESTRING_HASH),
//    serdes = options.getOrElse(StringConstants.SERDES, defaultSerdesType)
//            )


    String  storeType;

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

    public String getPartitioner() {
        return partitioner;
    }

    public void setPartitioner(String partitioner) {
        this.partitioner = partitioner;
    }

    public String getSerdes() {
        return serdes;
    }

    public void setSerdes(String serdes) {
        this.serdes = serdes;
    }

    String  namespace;
    String  name;
    String  path;
    int  totalPartitions;
    String  partitioner;
    String  serdes;

    String toPath(String delim){
        if (!StringUtils.isBlank(path)) {
            return path;
        } else {
           return  String.join(delim, storeType, namespace, name);
        }
    }

    @Override
    Meta.StoreLocator toProto() {

        Meta.StoreLocator.Builder   builder =  Meta.StoreLocator.newBuilder();

        return builder.setName(name)
        .setNamespace(namespace)
        .setPartitioner(partitioner)
        .setStoreType(storeType).
                setPath(path).
                setTotalPartitions(totalPartitions).
                setSerdes(serdes).build();


    }

    public   static  ErStoreLocator  parseFromPb(Meta.StoreLocator  storeLocator){


        ErStoreLocator  erStoreLocator = new  ErStoreLocator(storeLocator.getNamespace(),
                storeLocator.getName(),
                storeLocator.getPath(),
                storeLocator.getStoreType(),
                storeLocator.getTotalPartitions(),
                storeLocator.getPartitioner(),
                storeLocator.getSerdes()
        );
        return erStoreLocator;

    }


    public String  toString(){
        return JsonUtil.object2Json(this);
    }



}



