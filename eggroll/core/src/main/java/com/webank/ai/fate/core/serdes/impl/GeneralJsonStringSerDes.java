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

package com.webank.ai.fate.core.serdes.impl;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectWriter;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.fasterxml.jackson.databind.type.MapType;
import com.fasterxml.jackson.databind.type.TypeFactory;
import com.webank.ai.fate.core.serdes.SerDes;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.springframework.context.annotation.Scope;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.io.Serializable;
import java.util.HashMap;

@Component
@Scope("prototype")
public class GeneralJsonStringSerDes extends BaseSerDes implements SerDes {

    private static final MapType defaultHashMapType
            = TypeFactory.defaultInstance().constructMapType(HashMap.class, String.class, String.class);
    private static final Logger LOGGER = LogManager.getLogger(GeneralJsonStringSerDes.class);
    private final ObjectMapper objectMapper;
    private final ObjectWriter defaultWriter;
    private final ObjectWriter prettyWriter;
    private final ObjectMapper stringMapper;

    public GeneralJsonStringSerDes() {
        objectMapper = new ObjectMapper();

        objectMapper.configure(SerializationFeature.FAIL_ON_EMPTY_BEANS, false);
        objectMapper.configure(JsonParser.Feature.ALLOW_COMMENTS, true);
        objectMapper.configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);
        objectMapper.setSerializationInclusion(JsonInclude.Include.NON_NULL);

        objectMapper.enableDefaultTyping(ObjectMapper.DefaultTyping.OBJECT_AND_NON_CONCRETE);
        objectMapper.enableDefaultTyping(ObjectMapper.DefaultTyping.NON_CONCRETE_AND_ARRAYS);
        objectMapper.enableDefaultTyping(ObjectMapper.DefaultTyping.NON_FINAL);


        defaultWriter = objectMapper.writer();
        prettyWriter = objectMapper.writerWithDefaultPrettyPrinter();

        stringMapper = new ObjectMapper();
    }

    @Override
    public String serialize(Object object) {
        if (object == null) {
            return null;
        }
        super.checkParamType(Object.class, object.getClass());

        try {
            return defaultWriter.writeValueAsString(object);
        } catch (JsonProcessingException e) {
            throw new RuntimeException("Failed to serialize object to json string", e);
        }
    }

    @Override
    public <T> T deserialize(Serializable serializable, Class<T> clazz) {
        if (serializable == null) {
            return null;
        }

        super.checkParamType(String.class, serializable.getClass());

        try {
            return objectMapper.readValue((String) serializable, clazz);
        } catch (IOException e) {
            throw new RuntimeException("Failed to deserialize json byte array to " + clazz.getCanonicalName(), e);
        }
    }
}
