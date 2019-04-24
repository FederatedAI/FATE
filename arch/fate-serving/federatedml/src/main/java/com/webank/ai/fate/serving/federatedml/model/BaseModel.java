package com.webank.ai.fate.serving.federatedml.model;

import com.google.protobuf.ByteString;
import com.webank.ai.fate.api.networking.proxy.DataTransferServiceGrpc;
import com.webank.ai.fate.api.networking.proxy.Proxy;
import com.webank.ai.fate.core.network.grpc.client.ClientPool;
import com.webank.ai.fate.core.result.ReturnResult;
import com.webank.ai.fate.core.utils.Configuration;
import com.webank.ai.fate.core.utils.ObjectTransform;
import io.grpc.ManagedChannel;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public abstract class BaseModel {
    private static final Logger LOGGER = LogManager.getLogger();
    public abstract int initModel(byte[] protoMeta, byte[] protoParam);

    // public abstract HashMap<String, Object> predict(HashMap<String, Object> inputData);
    public abstract Map<String, Object> predict(Map<String, Object> inputData, Map<String, Object> predictParams);

    protected Map<String, Object> getFederatedPredict(Map<String, Object> predictParams) {
        Map<String, Object> requestData = new HashMap<>();
        requestData.put("sid", predictParams.get("sid"));
        requestData.put("partnerPartyId", predictParams.get("partyId"));
        requestData.put("partnerRole", predictParams.get("role"));
        requestData.put("partnerModelName", predictParams.get("modelName"));
        requestData.put("partnerModelNamespace", predictParams.get("modelNamespace"));
        requestData.put("allParty", predictParams.get("allParty"));
        //TODO:
        Map<String, List<Integer>> allParty = (Map<String, List<Integer>>)predictParams.get("allParty");
        requestData.put("role", "host");
        requestData.put("partyId", allParty.get("host").get(0));

        Proxy.Packet.Builder packetBuilder = Proxy.Packet.newBuilder();
        packetBuilder.setBody(Proxy.Data.newBuilder()
                .setValue(ByteString.copyFrom(ObjectTransform.bean2Json(requestData).getBytes()))
                .build());

        Proxy.Metadata.Builder metaDataBuilder = Proxy.Metadata.newBuilder();
        Proxy.Topic.Builder topicBuilder = Proxy.Topic.newBuilder();

        metaDataBuilder.setSrc(
                topicBuilder.setPartyId(requestData.get("partnerPartyId").toString()).
                        setRole("serving")
                        .setName("partnerPartyName")
                        .build());
        metaDataBuilder.setDst(
                topicBuilder.setPartyId(requestData.get("partyId").toString())
                        .setRole("serving")
                        .setName("partyName")
                        .build());
        metaDataBuilder.setCommand(Proxy.Command.newBuilder().setName("federatedPredict").build());
        metaDataBuilder.setConf(Proxy.Conf.newBuilder().setOverallTimeout(60*1000));
        packetBuilder.setHeader(metaDataBuilder.build());

        ManagedChannel channel1 = ClientPool.getChannel(Configuration.getProperty("proxy"));
        DataTransferServiceGrpc.DataTransferServiceBlockingStub stub1 = DataTransferServiceGrpc.newBlockingStub(channel1);
        Proxy.Packet packet = stub1.unaryCall(packetBuilder.build());
        ReturnResult result = (ReturnResult) ObjectTransform.json2Bean(packet.getBody().getValue().toStringUtf8(), ReturnResult.class);
        return result.getData();
    }
}
