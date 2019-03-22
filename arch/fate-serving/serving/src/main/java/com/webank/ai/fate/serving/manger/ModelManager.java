package com.webank.ai.fate.serving.manger;

import com.webank.ai.fate.core.mlmodel.model.MLModel;
import com.webank.ai.fate.core.mlmodel.buffer.ProtoModelBuffer;
import com.webank.ai.fate.core.statuscode.ReturnCode;
import com.webank.ai.fate.core.storage.kv.DTable;
import com.webank.ai.fate.core.utils.Configuration;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.commons.lang3.StringUtils;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

public class ModelManager {
    private ModelPool onlineModels;
    private ModelPool modelPool;
    private static final Logger LOGGER = LogManager.getLogger();
    private String modelPackage = "com.webank.ai.fate.serving.federatedml.model";

    public ModelManager(){
        this.onlineModels = new ModelPool();
        this.modelPool = new ModelPool();
    }


    public MLModel getModel(String sceneId, String partnerPartyId, String myRole){
        return this.onlineModels.get(this.getOnlineModelKey(sceneId, partnerPartyId, myRole));
    }

    public ProtoModelBuffer readModel(String sceneId, String partnerPartyId, String myRole, String commitId, String tag, String branch) throws Exception{
        String sceneKey = Version.getSceneKey(sceneId, Configuration.getProperty("partyId"), partnerPartyId, myRole);
        DTable dataTable = Version.getDTable("model_data", sceneKey, commitId, tag, branch);
        ProtoModelBuffer modelBuffer = new ProtoModelBuffer();
        modelBuffer.deserialize(dataTable.get("model_meta"), dataTable.get("model_param"), dataTable.get("data_transform"));
        return modelBuffer;
    }


    public MLModel loadModel(String sceneId, String partnerPartyId, String myRole, String commitId, String tag, String branch){
        try{
            ProtoModelBuffer modelBuffer = this.readModel(sceneId, partnerPartyId, myRole, commitId, tag, branch);
            Class modelClass = Class.forName(this.modelPackage + "." + modelBuffer.getMeta().getName());
            MLModel mlModel = (MLModel)modelClass.getConstructor().newInstance();
            Map<String, String> modelInfo = new HashMap<>();
            modelInfo.put("sceneId", sceneId);
            modelInfo.put("partnerPartyId", partnerPartyId);
            modelInfo.put("myRole", myRole);
            modelInfo.put("commitId", commitId);
            modelInfo.put("tag", tag);
            modelInfo.put("branch", branch);
            mlModel.setModelInfo(modelInfo);
            mlModel.initModel(modelBuffer);
            return mlModel;
        }
        catch (Exception ex){
            ex.printStackTrace();
            return null;
        }
    }

    private String getOnlineModelKey(String sceneId, String partnerPartyId, String myRole){
        String[] tmp = {sceneId, partnerPartyId, myRole};
        return StringUtils.join(tmp, "-");
    }

    private String genModelKey(String sceneId, String partnerPartyId, String myRole, String commitId){
        String[] tmp = {sceneId, partnerPartyId, myRole, commitId};
        return StringUtils.join(tmp, "-");
    }


    public String publishLoadModel(String sceneId, String partnerPartyId, String myRole, String commitId, String tag, String branch){
        this.modelPool.put(this.genModelKey(sceneId, partnerPartyId, myRole, commitId), this.loadModel(sceneId, partnerPartyId, myRole, commitId, tag, branch));
        return commitId;
    }

    public int publishOnlineModel(String sceneId, String partnerPartyId, String myRole, String commitId){
        MLModel model = this.modelPool.get(this.genModelKey(sceneId, partnerPartyId, myRole, commitId));
        if (model != null){
            this.onlineModels.put(this.getOnlineModelKey(sceneId, partnerPartyId, myRole), model);
            return ReturnCode.OK;
        }
        else{
            return ReturnCode.NOMODEL;
        }
    }
}
