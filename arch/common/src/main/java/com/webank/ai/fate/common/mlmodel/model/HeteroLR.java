package com.webank.ai.fate.common.mlmodel.model;

import com.webank.ai.fate.common.mlmodel.buffer.ModelBuffer;
import com.webank.ai.fate.common.mlmodel.buffer.ProtoModelBuffer;
import com.webank.ai.fate.common.statuscode.ReturnCode;
import org.apache.commons.lang3.ArrayUtils;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;

public abstract class HeteroLR extends BaseModel<ModelBuffer, float[], HashMap<String, Object>, HashMap<String, Object>>{
    protected float[] weight;
    protected HashMap<String, Object> meta;

    public void setWeight(float[] weight){
        this.weight = weight.clone();
    }

    public float[] getWeight(){
        return this.weight;
    }

    @Override
    public ModelBuffer export_model(){
        ArrayList<Float> weight = new ArrayList<>();
        Collections.addAll(weight, ArrayUtils.toObject(this.weight));
        ModelBuffer modelBuffer = new ProtoModelBuffer();
        modelBuffer.setMetaField("name", this.getClass().getName());
        modelBuffer.setParamField("weight", weight);
        return modelBuffer;
    }

    @Override
    public int init_model(ModelBuffer modelBuffer){
        List<Float> weight = (List<Float>) modelBuffer.getParamField("weight");
        this.weight = ArrayUtils.toPrimitive(weight.toArray(new Float[weight.size()]));
        return ReturnCode.OK;
    }

    @Override
    public abstract HashMap<String, Object> predict(float[] inputData, HashMap<String, Object> predictParams);
}
