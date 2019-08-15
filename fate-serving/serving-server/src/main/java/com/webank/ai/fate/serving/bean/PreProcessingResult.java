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

import java.util.Map;

public class PreProcessingResult {
    private Map<String, Object> featureIds;
    private Map<String, Object> processingResult;

    public Map<String, Object> getFeatureIds() {
        return featureIds;
    }

    public Map<String, Object> getProcessingResult() {
        return processingResult;
    }

    public void setProcessingResult(Map<String, Object> processingResult) {
        this.processingResult = processingResult;
    }

    public void setFeatureIds(Map<String, Object> featureIds) {
        this.featureIds = featureIds;
    }
}
