package com.webank.ai.fate.core.mlmodel.buffer;

import java.util.ArrayList;
import com.webank.ai.fate.core.mlmodel.buffer.ModelMetaProto.ModelMeta;
import com.webank.ai.fate.core.mlmodel.buffer.ModelParamProto.ModelParam;
import com.webank.ai.fate.core.mlmodel.buffer.DataTransformProto.DataTransform;
import com.webank.ai.fate.core.statuscode.ReturnCode;

public class ProtoModelBuffer{
    private ModelParam.Builder paramBuilder;
    private ModelMeta.Builder metaBuilder;
    private DataTransform.Builder dataTransformBuilder;
    private ModelParam param;
    private ModelMeta meta;
    private DataTransform dataTransform;

    public ProtoModelBuffer(){
        this.metaBuilder = ModelMeta.newBuilder();
        this.paramBuilder = ModelParam.newBuilder();
        this.dataTransformBuilder = DataTransform.newBuilder();
    }

    public ModelParam getParam() {
        return this.param;
    }

    public ModelMeta getMeta() {
        return this.meta;
    }

    public DataTransform getDataTransform() {
        return this.dataTransform;
    }

    public ArrayList<byte[]> serialize(){
        try {
            ArrayList<byte[]> bufferSteam = new ArrayList<>();
            bufferSteam.add(this.metaBuilder.build().toByteArray());
            bufferSteam.add(this.paramBuilder.build().toByteArray());
            bufferSteam.add(this.dataTransformBuilder.build().toByteArray());
            return bufferSteam;
        }
        catch (Exception ex){
            ex.printStackTrace();
            return null;
        }
    }

    public int deserialize(byte[] metaStream, byte[] paramStream, byte[] dataTransformStream){
        try{
            this.meta = ModelMeta.parseFrom(metaStream);
            this.param = ModelParam.parseFrom(paramStream);
            this.dataTransform = DataTransform.parseFrom(dataTransformStream);
            return ReturnCode.OK;
        }
        catch (Exception ex){
            ex.printStackTrace();
            return ReturnCode.UNKNOWNERROR;
        }
    }
}
