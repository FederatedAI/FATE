package com.webank.ai.fate.serving.federatedml.model;

import com.google.protobuf.ByteString;
import com.webank.ai.fate.api.networking.proxy.DataTransferServiceGrpc;
import com.webank.ai.fate.api.networking.proxy.Proxy;
import com.webank.ai.fate.core.bean.FederatedParty;
import com.webank.ai.fate.core.bean.FederatedRoles;
import com.webank.ai.fate.core.bean.ReturnResult;
import com.webank.ai.fate.core.network.grpc.client.ClientPool;
import com.webank.ai.fate.core.utils.Configuration;
import com.webank.ai.fate.core.utils.ObjectTransform;
import com.webank.ai.fate.serving.core.manager.CacheManager;
import io.grpc.ManagedChannel;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

public abstract class BaseModel {
    private static final Logger LOGGER = LogManager.getLogger();

    public abstract int initModel(byte[] protoMeta, byte[] protoParam);

    public abstract Map<String, Object> predict(Map<String, Object> inputData, Map<String, Object> predictParams);

    protected ReturnResult getFederatedPredict(Map<String, Object> federatedParams) {
        FederatedParty srcParty = (FederatedParty) federatedParams.get("local");
        FederatedRoles federatedRoles = (FederatedRoles) federatedParams.get("role");
        Map<String, Object> featureIds = (Map<String, Object>) federatedParams.get("feature_id");


        //TODO: foreach
        FederatedParty dstParty = new FederatedParty("host", federatedRoles.getRole("host").get(0));
        ReturnResult remoteResultFromCache = CacheManager.getRemoteModelInferenceResult(dstParty, federatedRoles, featureIds);
        if (remoteResultFromCache != null) {
            LOGGER.info("Get remote party model inference result from cache.");
            federatedParams.put("getRemotePartyResult", false);
            return remoteResultFromCache;
        }

        Map<String, Object> requestData = new HashMap<>();
        Arrays.asList("caseid", "seqno").forEach((field -> {
            requestData.put(field, federatedParams.get(field));
        }));
        requestData.put("partner_local", ObjectTransform.bean2Json(srcParty));
        requestData.put("partner_model_info", ObjectTransform.bean2Json(federatedParams.get("model_info")));
        requestData.put("feature_id", ObjectTransform.bean2Json(federatedParams.get("feature_id")));
        requestData.put("local", ObjectTransform.bean2Json(dstParty));
        requestData.put("role", ObjectTransform.bean2Json(federatedParams.get("role")));
        federatedParams.put("getRemotePartyResult", true);
        ReturnResult remoteResult = getFederatedPredictFromRemote(srcParty, dstParty, requestData);
        CacheManager.putRemoteModelInferenceResult(dstParty, federatedRoles, featureIds, remoteResult);
        LOGGER.info("Get remote party model inference result from federated request.");
        return remoteResult;
    }

    protected ReturnResult getFederatedPredictFromRemote(FederatedParty srcParty, FederatedParty dstParty, Map<String, Object> requestData) {

        Proxy.Packet.Builder packetBuilder = Proxy.Packet.newBuilder();
        packetBuilder.setBody(Proxy.Data.newBuilder()
                .setValue(ByteString.copyFrom(ObjectTransform.bean2Json(requestData).getBytes()))
                .build());

        Proxy.Metadata.Builder metaDataBuilder = Proxy.Metadata.newBuilder();
        Proxy.Topic.Builder topicBuilder = Proxy.Topic.newBuilder();

        metaDataBuilder.setSrc(
                topicBuilder.setPartyId(String.valueOf(srcParty.getPartyId())).
                        setRole("serving")
                        .setName("partnerPartyName")
                        .build());
        metaDataBuilder.setDst(
                topicBuilder.setPartyId(String.valueOf(dstParty.getPartyId()))
                        .setRole("serving")
                        .setName("partyName")
                        .build());
        metaDataBuilder.setCommand(Proxy.Command.newBuilder().setName("federatedInference").build());
        metaDataBuilder.setConf(Proxy.Conf.newBuilder().setOverallTimeout(60 * 1000));
        packetBuilder.setHeader(metaDataBuilder.build());

        ManagedChannel channel1 = ClientPool.getChannel(Configuration.getProperty("proxy"));
        DataTransferServiceGrpc.DataTransferServiceBlockingStub stub1 = DataTransferServiceGrpc.newBlockingStub(channel1);
        Proxy.Packet packet = stub1.unaryCall(packetBuilder.build());
        ReturnResult remoteResult = (ReturnResult) ObjectTransform.json2Bean(packet.getBody().getValue().toStringUtf8(), ReturnResult.class);
        return remoteResult;
    }
}
