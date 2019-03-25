package com.webank.ai.fate.core.mlmodel.buffer;

import java.util.ArrayList;
import com.webank.ai.fate.core.mlmodel.buffer.ModelMetaProto.ModelMeta;
import com.webank.ai.fate.core.mlmodel.buffer.ModelParamProto.ModelParam;
import com.webank.ai.fate.core.mlmodel.buffer.DataTransformServerProto.DataTransformServer;
import com.webank.ai.fate.core.result.StatusCode;

public class ProtoModelBuffer{
    private ModelParam.Builder paramBuilder;
    private ModelMeta.Builder metaBuilder;
    private DataTransformServer.Builder dataTransformServerBuilder;
    private ModelParam param;
    private ModelMeta meta;
    private DataTransformServer dataTransformServer;

    public ProtoModelBuffer(){
        this.metaBuilder = ModelMeta.newBuilder();
        this.paramBuilder = ModelParam.newBuilder();
        this.dataTransformServerBuilder = DataTransformServer.newBuilder();
    }

    public ModelParam getParam() {
        return this.param;
    }

    public ModelMeta getMeta() {
        return this.meta;
    }

    public DataTransformServer getDataTransformServer() {
        return this.dataTransformServer;
    }

    public ArrayList<byte[]> serialize(){
        try {
            ArrayList<byte[]> bufferSteam = new ArrayList<>();
            bufferSteam.add(this.metaBuilder.build().toByteArray());
            bufferSteam.add(this.paramBuilder.build().toByteArray());
            bufferSteam.add(this.dataTransformServerBuilder.build().toByteArray());
            return bufferSteam;
        }
        catch (Exception ex){
            ex.printStackTrace();
            return null;
        }
    }

    public int deserialize(byte[] metaStream, byte[] paramStream, byte[] dataTransformServerStream){
        try{
            this.meta = ModelMeta.parseFrom(metaStream);
            this.param = ModelParam.parseFrom(paramStream);
            this.dataTransformServer = DataTransformServer.parseFrom(dataTransformServerStream);
            return StatusCode.OK;
        }
        catch (Exception ex){
            ex.printStackTrace();
            return StatusCode.UNKNOWNERROR;
        }
    }
}
