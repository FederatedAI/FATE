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

package com.webank.ai.fate.core.factory;

import com.webank.ai.fate.api.core.BasicMeta;
import com.webank.ai.fate.core.constant.RuntimeConstants;
import com.webank.ai.fate.core.serdes.impl.ByteStringSerDesHelper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

@Component
// todo: combine this with serdes
public class CallMetaModelFactory {
    private static final String NO_ERROR = "NO_ERROR";
    private static final String HOST_LANGUAGE = "JAVA8";
    @Autowired
    private ByteStringSerDesHelper byteStringSerDesHelper;
    @Autowired
    private ReturnStatusFactory returnStatusFactory;

    public BasicMeta.CallResponse createNormalCallResponse(Object object) {
        BasicMeta.Data result = createData(object);
        BasicMeta.CallResponse.Builder builder = BasicMeta.CallResponse.newBuilder()
                .setReturnStatus(returnStatusFactory.create(0, NO_ERROR))
                .setResult(result);

        return builder.build();
    }

    public BasicMeta.CallResponse createErrorCallResponse(int errcode, String errmsg, Object object) {
        BasicMeta.Data result = createData(object);
        BasicMeta.CallResponse.Builder builder = BasicMeta.CallResponse.newBuilder()
                .setReturnStatus(returnStatusFactory.create(errcode, errmsg))
                .setResult(result);

        return builder.build();
    }

    public BasicMeta.Data createData(Object object) {
        BasicMeta.Data.Builder builder = BasicMeta.Data.newBuilder();

        if (object == null) {
            builder.setIsNull(true);
        } else {
            builder.setIsNull(false)
                    .setHostLanguage(HOST_LANGUAGE)
                    .setData(byteStringSerDesHelper.serialize(object))
                    .setType(object.getClass().getCanonicalName());
        }

        return builder.build();
    }

    public BasicMeta.CallRequest createCallRequestFromObject(Object object) {
        BasicMeta.CallRequest.Builder builder = BasicMeta.CallRequest.newBuilder();

        BasicMeta.Data.Builder dataBuilder = BasicMeta.Data.newBuilder();
        dataBuilder.setHostLanguage(RuntimeConstants.HOST_LANGUAGE)
                .setType(object.getClass()
                        .getCanonicalName())
                .setData(byteStringSerDesHelper.serialize(object));

        builder.setParam(dataBuilder.build());

        return builder.build();
    }

    // todo: 1. params check; 2. exception handling
    public Object extractModelObject(BasicMeta.CallResponse response) {
        Object result = null;
        if (response == null) {
            return result;
        }

        BasicMeta.Data data = response.getResult();

        if (data.getIsNull()) {
            return result;
        }

        try {
            Class<? extends Object> resultClass = Class.forName(data.getType());
            result = byteStringSerDesHelper.deserialize(data.getData(), resultClass);
        } catch (ClassNotFoundException e) {
            throw new RuntimeException(e);
        }

        return result;
    }
}
