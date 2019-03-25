package com.webank.ai.fate.serving.service;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.protobuf.ByteString;
import com.webank.ai.fate.api.networking.proxy.DataTransferServiceGrpc;
import com.webank.ai.fate.api.networking.proxy.Proxy;
import com.webank.ai.fate.api.networking.proxy.Proxy.Packet;
import com.webank.ai.fate.core.utils.Configuration;
import com.webank.ai.fate.serving.manger.ModelManager;
import io.grpc.stub.StreamObserver;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

public class ProxyService extends DataTransferServiceGrpc.DataTransferServiceImplBase {
    private static final Logger LOGGER = LogManager.getLogger();

    @Override
    public void unaryCall(Proxy.Packet req, StreamObserver<Proxy.Packet> responseObserver) {
        try {
            Map<String, Object> requestData = new ObjectMapper().readValue(req.getBody().getValue().toStringUtf8(), HashMap.class);
            Map<String, Object> responseData = new HashMap<>();

            switch (req.getHeader().getCommand().getName()) {
                case "federatedPredict":
                    responseData.putAll(new PredictService().federatedPredict(requestData));
                    break;
                case "federatedLoadModel":
                    responseData.putAll(new ModelManager().federatedLoadModel(requestData));
            }

            Packet.Builder packetBuilder = Packet.newBuilder();
            packetBuilder.setBody(Proxy.Data.newBuilder()
                    .setValue(ByteString.copyFrom(new ObjectMapper().writeValueAsString(responseData).getBytes()))
                    .build());

            Proxy.Metadata.Builder metaDataBuilder = Proxy.Metadata.newBuilder();
            Proxy.Topic.Builder topicBuilder = Proxy.Topic.newBuilder();

            metaDataBuilder.setSrc(
                    topicBuilder.setPartyId(Configuration.getProperty("partyId"))
                            .setRole("host")
                            .setName("myPartyName")
                            .build());
            metaDataBuilder.setDst(
                    topicBuilder.setPartyId((String) requestData.get("myPartyId"))
                            .setRole((String) requestData.get("myRole"))
                            .setName("partnerPartyName")
                            .build());
            packetBuilder.setHeader(metaDataBuilder.build());
            responseObserver.onNext(packetBuilder.build());
            responseObserver.onCompleted();
        } catch (IOException ex) {
            LOGGER.error(ex);
        }
    }
}