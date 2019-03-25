package com.webank.ai.fate.serving.manger;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.protobuf.ByteString;
import com.webank.ai.fate.api.networking.proxy.DataTransferServiceGrpc;
import com.webank.ai.fate.api.networking.proxy.Proxy;
import com.webank.ai.fate.core.mlmodel.model.MLModel;
import com.webank.ai.fate.core.mlmodel.buffer.ProtoModelBuffer;
import com.webank.ai.fate.core.network.grpc.client.ClientPool;
import com.webank.ai.fate.core.result.ReturnResult;
import com.webank.ai.fate.core.result.StatusCode;
import com.webank.ai.fate.core.storage.kv.DTable;
import com.webank.ai.fate.core.utils.Configuration;
import io.grpc.ManagedChannel;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.commons.lang3.StringUtils;

import java.io.IOException;
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

    public MLModel getModel(String sceneId, String partnerPartyId, String myRole, String commitId){
        return this.modelPool.get(this.genModelKey(sceneId, partnerPartyId, myRole, commitId));
    }

    public ProtoModelBuffer readModel(String sceneId, String partnerPartyId, String myRole, String commitId, String tag, String branch) throws Exception{
        String sceneKey = VersionControl.getSceneKey(sceneId, Configuration.getProperty("partyId"), partnerPartyId, myRole);
        DTable dataTable = VersionControl.getDTable("model_data", sceneKey, commitId, tag, branch);
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

    public ReturnResult publishLoadModel(String sceneId, String partnerPartyId, String myRole, String commitId, String tag, String branch){
        ReturnResult returnResult = new ReturnResult();
        returnResult.setStatusCode(StatusCode.OK);
        this.modelPool.put(this.genModelKey(sceneId, partnerPartyId, myRole, commitId), this.loadModel(sceneId, partnerPartyId, myRole, commitId, tag, branch));
        try{
            Map<String, Object> federatedLoadModelResult = this.requestFederatedLoadModel(sceneId, partnerPartyId, myRole, commitId, tag, branch);
        }
        catch (IOException ex){
            returnResult.setStatusCode(StatusCode.FEDERATEDERROR);
            returnResult.setError(ex.getMessage());
        }
        catch (Exception ex){
            returnResult.setStatusCode(StatusCode.FEDERATEDERROR);
            returnResult.setError(ex.getMessage());
        }
        returnResult.setData("commitId", commitId);
        return returnResult;
    }

    public Map<String, Object> federatedLoadModel(Map<String, Object> requestData){
        ReturnResult returnResult = new ReturnResult();
        returnResult.setStatusCode(StatusCode.OK);
        try{
            String sceneId = String.valueOf(requestData.get("sceneId"));
            String partnerPartyId = String.valueOf(requestData.get("partnerPartyId"));
            String myRole = String.valueOf(requestData.get("myRole"));
            String commitId = String.valueOf(requestData.get("commitId"));
            String branch = String.valueOf(requestData.get("branch"));
            String tag = String.valueOf(requestData.get("tag"));
            this.modelPool.put(this.genModelKey(sceneId, partnerPartyId, myRole, commitId), this.loadModel(sceneId, partnerPartyId, myRole, commitId, tag, branch));
            returnResult.setData("commitId", commitId);
        }
        catch (Exception ex){
            returnResult.setStatusCode(StatusCode.NOMODEL);
            returnResult.setMessage(ex.getMessage());
        }
        return ReturnResult.toMap(returnResult);
    }

    public int publishOnlineModel(String sceneId, String partnerPartyId, String myRole, String commitId){
        MLModel model = this.modelPool.get(this.genModelKey(sceneId, partnerPartyId, myRole, commitId));
        if (model != null){
            this.onlineModels.put(this.getOnlineModelKey(sceneId, partnerPartyId, myRole), model);
            return StatusCode.OK;
        }
        else{
            return StatusCode.NOMODEL;
        }
    }

    private Map<String, Object> requestFederatedLoadModel(String sceneId, String partnerPartyId, String myRole, String commitId, String tag, String branch) throws IOException {
        Proxy.Packet.Builder packetBuilder = Proxy.Packet.newBuilder();
        Map<String, String> requestData = new HashMap<>();
        requestData.put("sceneId", sceneId);
        requestData.put("myPartyId", Configuration.getProperty("partyId"));
        requestData.put("partnerPartyId", partnerPartyId);
        requestData.put("myRole", "guest");
        requestData.put("commitId", commitId);
        requestData.put("tag", tag);
        requestData.put("branch", branch);
        ObjectMapper objectMapper = new ObjectMapper();
        packetBuilder.setBody(Proxy.Data.newBuilder()
                .setValue(ByteString.copyFrom(objectMapper.writeValueAsString(requestData).getBytes()))
                .build());

        Proxy.Metadata.Builder metaDataBuilder = Proxy.Metadata.newBuilder();
        Proxy.Topic.Builder topicBuilder = Proxy.Topic.newBuilder();

        metaDataBuilder.setSrc(
                topicBuilder.setPartyId(requestData.get("partnerPartyId"))
                        .setRole(myRole)
                        .setName("myPartyName")
                        .build());
        metaDataBuilder.setDst(
                topicBuilder.setPartyId(partnerPartyId)
                        .setRole("host")
                        .setName("partnerPartyName")
                        .build());
        metaDataBuilder.setCommand(Proxy.Command.newBuilder().setName("publishLoadModel").build());
        packetBuilder.setHeader(metaDataBuilder.build());

        ManagedChannel channel1 = ClientPool.getChannel(Configuration.getProperty("proxy"));
        DataTransferServiceGrpc.DataTransferServiceBlockingStub stub1 = DataTransferServiceGrpc.newBlockingStub(channel1);
        Proxy.Packet packet = stub1.unaryCall(packetBuilder.build());
        return objectMapper.readValue(packet.getBody().getValue().toStringUtf8(), HashMap.class);
    }
}
