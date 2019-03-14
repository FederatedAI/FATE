package com.webank.ai.fate.common.mlmodel.manager;

import com.webank.ai.fate.common.mlmodel.buffer.ModelBuffer;
import com.webank.ai.fate.common.mlmodel.model.MLModel;
import com.webank.ai.fate.common.storage.kv.LocalFileKVPool;
import com.webank.ai.fate.common.mlmodel.buffer.ProtoModelBuffer;
import com.webank.ai.fate.common.utils.Configuration;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.commons.lang3.StringUtils;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.UUID;

public class ModelManager {
    private LocalFileKVPool storage;
    private ModelPool modelPool;
    private static final Logger LOGGER = LogManager.getLogger();

    public ModelManager(){
        this.modelPool = new ModelPool();
        this.storage = new LocalFileKVPool();
    }

    public void updatePool(){
        ArrayList<String[]> modelInfo = this.getAllModelInfo();
        modelInfo.forEach((item)->{
            this.modelPool.put(getModelKey(item[0], item[1], item[2]), this.loadModel(item[4]));
        });
    }

    public MLModel getModel(String role, String name){
        LOGGER.info(Configuration.getProperties());
        return this.modelPool.get(this.getModelKey(role, Configuration.getProperty(String.format("%s.partyId", role)), name));
    }

    public String saveModel(ModelBuffer modelBuffer){
        ArrayList<byte[]> bufferStream = modelBuffer.serialize();
        String modelId = UUID.randomUUID().toString().replace("-", "");
        this.storage.put(String.format("%s.meta", modelId), bufferStream.get(0));
        this.storage.put(String.format("%s.param", modelId), bufferStream.get(1));
        return modelId;
    }

    public ModelBuffer readModel(String modelId){
        byte[] metaStream = this.storage.get(String.format("%s.meta", modelId));
        byte[] paramStream = this.storage.get(String.format("%s.param", modelId));
        ProtoModelBuffer modelBuffer = new ProtoModelBuffer();
        modelBuffer.deserialize(metaStream, paramStream);
        return modelBuffer;
    }

    public MLModel loadModel(String modelId){
        try{
            ModelBuffer modelBuffer = this.readModel(modelId);
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

    private String getModelKey(String role, String partyId, String name){
        LOGGER.info(StringUtils.join(role, partyId, name));
        return StringUtils.join(role, partyId, name);
    }

    private ArrayList<String[]> getAllModelInfo(){
        ArrayList<String[]> modelInfo = new ArrayList<>();
        // query from mysql by host.partyId and guest.partId
        String[] tmp1 = {"host", "100001", "HeteroLRGuest", "HeteroLR", "7a067108a2ba44358bb3c70a001d5152"};
        modelInfo.add(tmp1);
        String[] tmp2 = {"guest", "100001", "HeteroLRHost", "HeteroLR", "9fabf840fa4a4c3e88b2a3f634bdd6c8"};
        modelInfo.add(tmp2);
        return modelInfo;
    }


    public static void main(String[] args){
        ModelManager modelManager = new ModelManager();
        modelManager.updatePool();
        /*

        //save guest
        HeteroLRGuest heteroLRGuest = new HeteroLRGuest();
        float[] weight1 = {1, 2};
        heteroLRGuest.setWeight(weight1);
        ModelBuffer modelBuffer1 = heteroLRGuest.export_model();
        String guestModelId = modelManager.saveModel(modelBuffer1);
        LOGGER.info(guestModelId);

        //host
        HeteroLRHost heteroLRHost = new HeteroLRHost();
        float[] weight2 = {2, 4};
        heteroLRHost.setWeight(weight2);
        ModelBuffer modelBuffer2 = heteroLRHost.export_model();
        String hostModelId = modelManager.saveModel(modelBuffer2);
        LOGGER.info(hostModelId);
        */

        //get
        //MLModel mlModel1 = modelManager.loadModel(guestModelId);
        MLModel mlModel1 = modelManager.getModel("guest", "HeteroLRGuest");
        float[] inputData = {10, 20};
        HashMap<String, String> param = new HashMap<>();
        param.put("p1", "ddd");
        LOGGER.info(mlModel1.predict(inputData, param));

        //MLModel mlModel2 = modelManager.loadModel(hostModelId);
        MLModel mlModel2 = modelManager.getModel("host", "HeteroLRHost");
        LOGGER.info(mlModel2.predict(inputData, param));

    }
}
