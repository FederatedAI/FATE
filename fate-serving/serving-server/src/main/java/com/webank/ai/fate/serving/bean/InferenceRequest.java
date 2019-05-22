/*
 * Copyright 2019 The FATE Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.webank.ai.fate.serving.bean;

import com.webank.ai.fate.serving.utils.InferenceUtils;

import java.util.HashMap;
import java.util.Map;

public class InferenceRequest {
    private int sceneid;
    private String modelName;
    private String modelNamespace;
    private String seqno;
    private String caseid;
    private Map<String, Object> featureData;
    InferenceRequest(){
        sceneid = 0;
        seqno = InferenceUtils.generateSeqno();
        caseid = InferenceUtils.generateCaseid();
        featureData = new HashMap<>();
    }

    public String getSeqno() {
        return seqno;
    }

    public int getSceneid() {
        return sceneid;
    }

    public String getCaseid() {
        return caseid;
    }

    public String getModelName() {
        return modelName;
    }

    public String getModelNamespace() {
        return modelNamespace;
    }

    public Map<String, Object> getFeatureData() {
        return featureData;
    }
}
