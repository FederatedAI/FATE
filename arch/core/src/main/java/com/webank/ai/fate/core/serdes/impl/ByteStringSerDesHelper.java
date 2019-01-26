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

import com.google.protobuf.ByteString;
import com.webank.ai.fate.core.serdes.SerDes;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.io.Serializable;

@Component
public class ByteStringSerDesHelper implements SerDes {
    @Autowired
    private GeneralJsonBytesSerDes generalJsonBytesSerDes;

    @Override
    public ByteString serialize(Object object) {
        ByteString result = ByteString.copyFrom(generalJsonBytesSerDes.serialize(object));

        return result;
    }

    @Override
    public Object deserialize(Serializable serializable) {
        return null;
    }

    @Override
    public <T> T deserialize(Serializable serializable, Class<T> clazz) {
        ByteString bs = null;

        if (serializable instanceof ByteString) {
            bs = (ByteString) serializable;
        } else {
            throw new IllegalArgumentException("serializable is not from type ByteString");
        }

        T result = generalJsonBytesSerDes.deserialize(bs.toByteArray(), clazz);

        return result;
    }
}
