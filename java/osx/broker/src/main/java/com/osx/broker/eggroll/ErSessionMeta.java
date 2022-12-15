package com.osx.broker.eggroll;


import com.osx.core.utils.JsonUtil;
import com.webank.eggroll.core.meta.Meta;

import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public class ErSessionMeta extends BaseProto<Meta.SessionMeta> {
    public String getId() {
        return id;
    }

    public void setId(String id) {
        this.id = id;
    }

    public String getStatus() {
        return status;
    }

    public void setStatus(String status) {
        this.status = status;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public int getTotalProcCount() {
        return totalProcCount;
    }

    public void setTotalProcCount(int totalProcCount) {
        this.totalProcCount = totalProcCount;
    }

    public int getActiveProcCount() {
        return activeProcCount;
    }

    public void setActiveProcCount(int activeProcCount) {
        this.activeProcCount = activeProcCount;
    }

    public String getTag() {
        return tag;
    }

    public void setTag(String tag) {
        this.tag = tag;
    }

    public List<ErProcessor> getProcessors() {
        return processors;
    }

    public void setProcessors(List<ErProcessor> processors) {
        this.processors = processors;
    }

    public Map<String, String> getOptions() {
        return options;
    }

    public void setOptions(Map<String, String> options) {
        this.options = options;
    }

    String id;
    String status;
    String name;
    int totalProcCount;
    int activeProcCount;
    String tag;
    List<ErProcessor> processors;
    Map<String,String> options;



    public Meta.SessionMeta   toProto(){
        Meta.SessionMeta.Builder        builder = Meta.SessionMeta.newBuilder()
                .setId(this.id)
                .setName(this.name)
                .setStatus(this.status)
                .setTag(this.tag)
                .addAllProcessors(this.processors.stream().map(ErProcessor::toProto).collect(Collectors.toList()))
                .putAllOptions(this.options);

        return  builder.build();
    }


    static ErSessionMeta parseFromPb( Meta.SessionMeta  sessionMeta) {
        if(sessionMeta==null){
            return null;
        }

        ErSessionMeta  erSessionMeta  = new ErSessionMeta();
        erSessionMeta.id = sessionMeta.getId();
        erSessionMeta.status = sessionMeta.getStatus();
        erSessionMeta.name = sessionMeta.getName();
        erSessionMeta.tag = sessionMeta.getTag();
        erSessionMeta.options = sessionMeta.getOptionsMap();
        erSessionMeta.processors = sessionMeta.getProcessorsList().stream().map(ErProcessor::parseFromPb).collect(Collectors.toList());
        return  erSessionMeta;
    }

    public  String toString(){
        return JsonUtil.object2Json(this);
    }
}



//        override def toProto[T >: PbMessage](): Meta.SessionMeta = {
//                val builder = Meta.SessionMeta.newBuilder()
//                .setId(src.id)
//                .setName(src.name)
//                .setStatus(src.status)
//                .setTag(src.tag)
//                .addAllProcessors(src.processors.toList.map(_.toProto()).asJava)
//                .putAllOptions(src.options.asJava)
//
//                builder.build()
//        }

//case class ErSessionMeta(id: String = StringConstants.EMPTY,
//                         name: String = StringConstants.EMPTY,
//                         status: String = StringConstants.EMPTY,
//                         totalProcCount: Int = 0,
//                         activeProcCount: Int = 0,
//                         tag: String = StringConstants.EMPTY,
//                         processors: Array[ErProcessor] = Array(),
//        options: Map[String, String] = Map()) extends MetaRpcMessage {
//        }

