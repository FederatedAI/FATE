package com.webank.ai.fate.core.mlmodel.model;


import java.util.HashMap;
import java.util.Map;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.protobuf.ByteString;
import com.webank.ai.fate.api.networking.proxy.Proxy;
import com.webank.ai.fate.core.network.grpc.client.ClientPool;
import com.webank.ai.fate.core.utils.Configuration;
import io.grpc.ManagedChannel;
import com.webank.ai.fate.api.networking.proxy.Proxy.Packet;
import com.webank.ai.fate.api.networking.proxy.DataTransferServiceGrpc;

public abstract class BaseModel<B, X, P> implements MLModel<B, X, P>{
    private Map<String, String> modelInfo;

    @Override
    public void setModelInfo(Map<String, String> modelInfo) {
        this.modelInfo = modelInfo;
    }

    @Override
    public Map<String, String> getModelInfo() {
        return this.modelInfo;
    }

    public String getModelId(){
        return this.modelInfo.get("modelId");
    }

    @Override
    public abstract int initModel(B modelBuffer);

    @Override
    public abstract Map<String, Object> predict(X inputData, P predictParams);

    protected Map<String, Object> getFederatedPredict(Map<String, Object> requestData) throws Exception{
        Packet.Builder packetBuilder = Packet.newBuilder();
        requestData.putAll(this.modelInfo);
        requestData.put("myPartyId", Configuration.getProperty("partyId"));
        ObjectMapper objectMapper = new ObjectMapper();
        packetBuilder.setBody(Proxy.Data.newBuilder()
                        .setValue(ByteString.copyFrom(objectMapper.writeValueAsString(requestData).getBytes()))
                        .build());

        Proxy.Metadata.Builder metaDataBuilder = Proxy.Metadata.newBuilder();
        Proxy.Topic.Builder topicBuilder = Proxy.Topic.newBuilder();

        metaDataBuilder.setSrc(
                topicBuilder.setPartyId(Configuration.getProperty("partyId")).
                        setRole(this.modelInfo.get("myRole"))
                        .setName("partyName")
                        .build());
        metaDataBuilder.setDst(
                topicBuilder.setPartyId(this.modelInfo.get("partnerPartyId"))
                        .setRole("host")
                        .setName("partnerPartyName")
                        .build());
        metaDataBuilder.setCommand(Proxy.Command.newBuilder().setName("federatedPredict").build());
        packetBuilder.setHeader(metaDataBuilder.build());

        ManagedChannel channel1 = ClientPool.getChannel(Configuration.getProperty("proxy"));
        DataTransferServiceGrpc.DataTransferServiceBlockingStub stub1 = DataTransferServiceGrpc.newBlockingStub(channel1);
        Packet packet = stub1.unaryCall(packetBuilder.build());

        Map<String, Object> result = objectMapper.readValue(packet.getBody().getValue().toStringUtf8(), HashMap.class);
        return result;
    }
}
