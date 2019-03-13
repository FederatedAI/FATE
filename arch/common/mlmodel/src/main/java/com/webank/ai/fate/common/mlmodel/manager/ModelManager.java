package com.webank.ai.fate.common.mlmodel.manager;

import com.webank.ai.fate.common.mlmodel.model.HeteroLRGuest;
import com.webank.ai.fate.core.statuscode.ReturnCode;
import com.webank.ai.fate.common.mlmodel.buffer.ModelBuffer;
import com.webank.ai.fate.common.mlmodel.model.MLModel;
import com.webank.ai.fate.common.storage.kv.LocalFileKVPool;
import com.webank.ai.fate.common.mlmodel.buffer.ProtoModelBuffer;

import java.util.ArrayList;
import java.util.HashMap;

public class ModelManager {
    private LocalFileKVPool storage;
    public ModelManager(){
        this.storage = new LocalFileKVPool();
    }

    public int saveModel(ModelBuffer modelBuffer){
        ArrayList<byte[]> bufferStream = modelBuffer.serialize();
        this.storage.put("meta", bufferStream.get(0));
        this.storage.put("param", bufferStream.get(1));
        return ReturnCode.OK;
    }

    public ModelBuffer readModel(String name, String nameSpace, String modelId){
        byte[] metaStream = this.storage.get("meta");
        byte[] paramStream = this.storage.get("param");
        ProtoModelBuffer modelBuffer = new ProtoModelBuffer();
        modelBuffer.deserialize(metaStream, paramStream);
        return modelBuffer;
    }

    public MLModel loadModel(String name, String nameSpace, String modelId){
        try{
            ModelBuffer modelBuffer = this.readModel(name, nameSpace, modelId);
            Class modelClass = Class.forName((String)modelBuffer.getMetaField("name"));
            MLModel mlModel = (MLModel)modelClass.getConstructor().newInstance();
            mlModel.init_model(modelBuffer);
            return mlModel;
        }
        catch (Exception ex){
            ex.printStackTrace();
            return null;
        }
    }

    public void updatePool(){

    }

    public static void main(String[] args){
        ModelManager modelManager = new ModelManager();

        //save
        HeteroLRGuest heteroLRGuest = new HeteroLRGuest();
        float[] weight = {1, 2};
        heteroLRGuest.setWeight(weight);
        ModelBuffer modelBuffer1 = heteroLRGuest.export_model();
        modelManager.saveModel(modelBuffer1);

        //get
        MLModel mlModel = modelManager.loadModel("1", "2", "3");
        float[] inputData = {10, 20};
        HashMap<String, String> param = new HashMap<>();
        param.put("p1", "ddd");
        System.out.println(mlModel.predict(inputData, param));
    }
}
