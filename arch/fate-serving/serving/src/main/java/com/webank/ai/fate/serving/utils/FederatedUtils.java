package com.webank.ai.fate.serving.utils;

import com.webank.ai.fate.api.serving.PredictionServiceProto.FederatedMeta;
import com.webank.ai.fate.core.utils.Configuration;

public class FederatedUtils {
    public static FederatedMeta.Builder genResponseMetaBuilder(FederatedMeta requestMeta, String modelId){
        FederatedMeta.Builder responseMetaBuilder = FederatedMeta.newBuilder();
        responseMetaBuilder.setPartyId(Configuration.getProperty("partyId"));
        responseMetaBuilder.setSceneId(requestMeta.getSceneId());
        responseMetaBuilder.setModelId(modelId);
        responseMetaBuilder.setRole(getMyRole(requestMeta.getRole()));
        return responseMetaBuilder;
    }

    public static String getMyRole(String requestRole) {
        String myRole;
        switch (requestRole) {
            case "guestUser":
                myRole = "guest";
                break;
            case "guest":
                myRole = "host";
                break;
            default:
                myRole = "unknown";
        }
        return myRole;
    }
}
