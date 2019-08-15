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

package com.webank.ai.fate.serving.adapter.dataaccess;

import com.webank.ai.fate.core.bean.ReturnResult;
import com.webank.ai.fate.serving.core.constant.InferenceRetCode;
import org.apache.commons.lang3.StringUtils;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class TestFile implements FeatureData {
    private static final Logger LOGGER = LogManager.getLogger();

    @Override
    public ReturnResult getData(Map<String, Object> featureIds) {
        ReturnResult returnResult = new ReturnResult();
        Map<String, Object> data = new HashMap<>();
        try {
            List<String> lines = Files.readAllLines(Paths.get(System.getProperty("user.dir"), "host_data.csv"));
            lines.forEach(line -> {
                for (String kv : StringUtils.split(line, ",")) {
                    String[] a = StringUtils.split(kv, ":");
                    data.put(a[0], Double.valueOf(a[1]));
                }
            });
            returnResult.setData(data);
            returnResult.setRetcode(InferenceRetCode.OK);
        } catch (Exception ex) {
            LOGGER.error(ex);
            returnResult.setRetcode(InferenceRetCode.GET_FEATURE_FAILED);
        }
        return returnResult;
    }
}
