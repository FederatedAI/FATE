package com.webank.ai.fate.serving.service;

import com.webank.ai.fate.api.mlmodel.manager.ModelServiceGrpc;
import com.webank.ai.fate.api.mlmodel.manager.ModelServiceProto.PublishRequest;
import com.webank.ai.fate.api.mlmodel.manager.ModelServiceProto.PublishResponse;
import com.webank.ai.fate.serving.manger.ModelManager;
import io.grpc.stub.StreamObserver;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

public class ModelService extends ModelServiceGrpc.ModelServiceImplBase{
    private static final Logger LOGGER = LogManager.getLogger();

    @Override
    public void publishLoad(PublishRequest req, StreamObserver<PublishResponse> responseStreamObserver){
        PublishResponse.Builder builder = PublishResponse.newBuilder();
        int loadStatus = new ModelManager().publishLoadModel(req.getModelId());
        builder.setStatusCode(loadStatus);
        builder.setModelId(req.getModelId());
        responseStreamObserver.onNext(builder.build());
        responseStreamObserver.onCompleted();
    }
    @Override
    public void publishOnline(PublishRequest req, StreamObserver<PublishResponse> responseStreamObserver) {
        PublishResponse.Builder builder = PublishResponse.newBuilder();
        int onlineStatus = new ModelManager().publishOnlineModel(req.getSceneId(), req.getPartnerPartyId(), req.getMyRole(), req.getModelId());
        builder.setStatusCode(onlineStatus);
        responseStreamObserver.onNext(builder.build());
        responseStreamObserver.onCompleted();
    }
}
