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

import com.webank.ai.fate.core.constant.ExceptionConstants;
import com.webank.ai.fate.core.serdes.SerDes;

import java.io.Serializable;

public abstract class BaseSerDes implements SerDes {
    protected UnsupportedOperationException unsupportedOperationException;

    public BaseSerDes() {
        unsupportedOperationException = new UnsupportedOperationException("operation not supported");
    }

    @Override
    public Serializable serialize(Object object) {
        throw ExceptionConstants.unsupportedOperationException;
    }

    @Override
    public Object deserialize(Serializable serializable) {
        throw ExceptionConstants.unsupportedOperationException;
    }

    @Override
    public <T> T deserialize(Serializable serializable, Class<T> clazz) {
        throw ExceptionConstants.unsupportedOperationException;
    }

    public void checkParamType(Class expectedType, Class actualType) {
        if (!expectedType.isAssignableFrom(actualType)) {
            StringBuilder sb = new StringBuilder();
            sb.append("Type error in SerDes. Expected type: ")
                    .append(expectedType.getCanonicalName())
                    .append(", actual type: ")
                    .append(actualType.getCanonicalName());
            throw new IllegalArgumentException(sb.toString());
        }
    }
}
