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

package com.webank.ai.fate.core.utils;

import com.google.protobuf.Message;
import com.google.protobuf.util.JsonFormat;
import com.webank.ai.fate.core.serdes.impl.GeneralJsonStringSerDes;
import org.apache.commons.lang3.exception.ExceptionUtils;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Scope;
import org.springframework.stereotype.Component;


@Component
@Scope("prototype")
public class ToStringUtils {
    private static final Logger LOGGER = LogManager.getLogger(ToStringUtils.class);
    @Autowired
    private GeneralJsonStringSerDes jsonSerDes;
    private JsonFormat.Printer protoPrinter = JsonFormat.printer()
            .preservingProtoFieldNames()
            .includingDefaultValueFields()
            .omittingInsignificantWhitespace();

    public String toOneLineString(Message target) {
        String result = "[null]";

        if (target == null) {
            LOGGER.info("target is null");
            return result;
        }

        try {
            result = protoPrinter.print(target);
        } catch (Exception e) {
            LOGGER.info(ExceptionUtils.getStackTrace(e));
        }

        return result;
    }

    public String toOneLineString(Object object) {
        return jsonSerDes.serialize(object);
    }
}
