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

package com.webank.ai.fate.serving.core.utils;

import com.webank.ai.fate.core.mlmodel.buffer.DefaultEmptyFillProto;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

public class ProtobufUtils {
    private static final Logger LOGGER = LogManager.getLogger();

    public static <T> T parseProtoObject(com.google.protobuf.Parser<T> protoParser, byte[] protoString) throws com.google.protobuf.InvalidProtocolBufferException {
        T messageV3;
        try {
            messageV3 = protoParser.parseFrom(protoString);
            LOGGER.info("parse {} proto object normal", messageV3.getClass().getSimpleName());
            return messageV3;
        } catch (Exception ex1) {
            try {
                DefaultEmptyFillProto.DefaultEmptyFillMessage defaultEmptyFillMessage = DefaultEmptyFillProto.DefaultEmptyFillMessage.parseFrom(protoString);
                messageV3 = protoParser.parseFrom(new byte[0]);
                LOGGER.info("parse {} proto object with default values", messageV3.getClass().getSimpleName());
                return messageV3;
            } catch (Exception ex2) {
                throw ex1;
            }
        }
    }
}
