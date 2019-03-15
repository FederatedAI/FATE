package com.webank.ai.fate.common.mlmodel.buffer;

import java.util.ArrayList;
import com.webank.ai.fate.common.mlmodel.buffer.ModelMetaProto.ModelMeta;
import com.webank.ai.fate.common.mlmodel.buffer.ModelParamProto.ModelParam;
import com.webank.ai.fate.common.statuscode.ReturnCode;

public class ProtoModelBuffer{
    private ModelParam.Builder paramBuilder;
    private ModelMeta.Builder metaBuilder;
    private ModelParam param;
    private ModelMeta meta;

    public ProtoModelBuffer(){
        this.metaBuilder = ModelMeta.newBuilder();
        this.paramBuilder = ModelParam.newBuilder();
    }

    public ModelParam getParam() {
        return param;
    }

    public ModelMeta getMeta() {
        return meta;
    }

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
