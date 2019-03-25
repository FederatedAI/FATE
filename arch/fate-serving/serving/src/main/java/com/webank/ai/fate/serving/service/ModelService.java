package com.webank.ai.fate.serving.service;

import com.webank.ai.fate.api.mlmodel.manager.ModelServiceGrpc;
import com.webank.ai.fate.api.mlmodel.manager.ModelServiceProto.PublishRequest;
import com.webank.ai.fate.api.mlmodel.manager.ModelServiceProto.PublishResponse;
import com.webank.ai.fate.core.result.ReturnResult;
import com.webank.ai.fate.core.result.StatusCode;
import com.webank.ai.fate.core.utils.Configuration;
import com.webank.ai.fate.serving.manger.ModelManager;
import io.grpc.stub.StreamObserver;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

public class ModelService extends ModelServiceGrpc.ModelServiceImplBase{
    private static final Logger LOGGER = LogManager.getLogger();

    @Override
    public void publishLoad(PublishRequest req, StreamObserver<PublishResponse> responseStreamObserver){
        PublishResponse.Builder builder = PublishResponse.newBuilder();
        int loadStatus;
        LOGGER.info(Configuration.getProperty("partyId"));
        LOGGER.info(req.getMyPartyId());
        if (Configuration.getProperty("partyId").equals(req.getMyPartyId())){
            ReturnResult returnResult = new ModelManager().publishLoadModel(req.getSceneId(), req.getPartnerPartyId(), req.getMyRole(), req.getCommitId(), req.getTag(), req.getBranch());
            loadStatus = returnResult.getStatusCode();
            builder.setCommitId((String)returnResult.getData().get("commitId"));
        }
        else{
            loadStatus = StatusCode.NOTME;
        }
        builder.setStatusCode(loadStatus);
        responseStreamObserver.onNext(builder.build());
        responseStreamObserver.onCompleted();
    }

    @Override
    public void publishOnline(PublishRequest req, StreamObserver<PublishResponse> responseStreamObserver) {
        PublishResponse.Builder builder = PublishResponse.newBuilder();
        int onlineStatus = new ModelManager().publishOnlineModel(req.getSceneId(), req.getPartnerPartyId(), req.getMyRole(), req.getCommitId());
        builder.setStatusCode(onlineStatus);
        responseStreamObserver.onNext(builder.build());
        responseStreamObserver.onCompleted();
    }
}
