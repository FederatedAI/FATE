package com.webank.ai.fate.common.mlmodel.buffer;

import java.util.ArrayList;
import com.google.protobuf.GeneratedMessageV3;
import com.webank.ai.fate.common.mlmodel.buffer.ModelMetaProto.ModelMeta;
import com.webank.ai.fate.common.mlmodel.buffer.ModelParamProto.ModelParam;
import com.webank.ai.fate.common.statuscode.ReturnCode;

public class ProtoModelBuffer extends BaseModelBuffer<String, Object>{
    private ModelParam.Builder paramBuilder;
    private ModelMeta.Builder metaBuilder;
    private ModelParam param;
    private ModelMeta meta;

    public ProtoModelBuffer(){
        this.metaBuilder = ModelMeta.newBuilder();
        this.paramBuilder = ModelParam.newBuilder();
    }

    private void setField(GeneratedMessageV3.Builder builder, String name, Object value){
        builder.setField(builder.getDescriptorForType().findFieldByName(name), value);
    }

    @Override
    public void setMetaField(String name, Object value){
        this.setField(this.metaBuilder, name, value);
    }

    @Override
    public void setParamField(String name, Object value){
        this.setField(this.paramBuilder, name, value);
    }

    private Object getField(GeneratedMessageV3 message, String name){
        Object value = message.getField(message.getDescriptorForType().findFieldByName(name));
        return value;
    }

    @Override
    public Object getMetaField(String name){
        return getField(this.meta, name);
    }

    @Override
    public Object getParamField(String name){
        return getField(this.param, name);
    }

    @Override
    public ArrayList<byte[]> serialize(){
        try {
            ArrayList<byte[]> bufferSteam = new ArrayList<>();
            bufferSteam.add(this.metaBuilder.build().toByteArray());
            bufferSteam.add(this.paramBuilder.build().toByteArray());
            return bufferSteam;
        }
        catch (Exception ex){
            ex.printStackTrace();
            return null;
        }
    }

    @Override
    public int deserialize(byte[] metaStream, byte[] paramStream){
        try{
            this.meta = ModelMetaProto.ModelMeta.parseFrom(metaStream);
            this.param = ModelParamProto.ModelParam.parseFrom(paramStream);
            return ReturnCode.OK;
        }
        catch (Exception ex){
            ex.printStackTrace();
            return ReturnCode.RUNTIMEERROR;
        }
    }
}
