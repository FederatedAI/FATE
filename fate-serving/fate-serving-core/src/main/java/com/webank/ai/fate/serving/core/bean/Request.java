package com.webank.ai.fate.serving.core.bean;

import java.util.Map;

public interface Request {

    public String getSeqno();

    public String getAppid();

    public String getCaseid();

    public String getPartyId();

    public String getModelVersion();

    public String getModelId();

    public Map<String, Object> getFeatureData();
}
