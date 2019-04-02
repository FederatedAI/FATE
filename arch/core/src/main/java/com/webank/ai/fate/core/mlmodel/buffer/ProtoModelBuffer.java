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

package com.webank.ai.fate.core.mlmodel.buffer;

import java.io.IOException;
import java.util.ArrayList;
import com.webank.ai.fate.core.mlmodel.buffer.ModelMetaProto.ModelMeta;
import com.webank.ai.fate.core.mlmodel.buffer.ModelParamProto.ModelParam;
import com.webank.ai.fate.core.mlmodel.buffer.DataTransformServerProto.DataTransformServer;
import com.webank.ai.fate.core.constant.StatusCode;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

public class ProtoModelBuffer{
    private static final Logger LOGGER = LogManager.getLogger();
    private ModelParam.Builder paramBuilder;
    private ModelMeta.Builder metaBuilder;
    private DataTransformServer.Builder dataTransformServerBuilder;
    private ModelParam param;
    private ModelMeta meta;
    private DataTransformServer dataTransformServer;

    public ProtoModelBuffer(){
        this.metaBuilder = ModelMeta.newBuilder();
        this.paramBuilder = ModelParam.newBuilder();
        this.dataTransformServerBuilder = DataTransformServer.newBuilder();
    }

    public ModelParam getParam() {
        return this.param;
    }

    public ModelMeta getMeta() {
        return this.meta;
    }

    public DataTransformServer getDataTransformServer() {
        return this.dataTransformServer;
    }

    public ArrayList<byte[]> serialize(){
        try {
            ArrayList<byte[]> bufferSteam = new ArrayList<>();
            bufferSteam.add(this.metaBuilder.build().toByteArray());
            bufferSteam.add(this.paramBuilder.build().toByteArray());
            bufferSteam.add(this.dataTransformServerBuilder.build().toByteArray());
            return bufferSteam;
        }
        catch (Exception ex){
            LOGGER.error("Protobuffer serialize error: {}", ex.getMessage());
            return null;
        }
    }

    public int deserialize(byte[] metaStream, byte[] paramStream, byte[] dataTransformServerStream){
        if (metaStream == null || metaStream.length == 0 ||
                paramStream == null || paramStream.length == 0 ||
                dataTransformServerStream == null || dataTransformServerStream.length == 0){
            return StatusCode.PARAMERROR;
        }
        try{
            this.meta = ModelMeta.parseFrom(metaStream);
            this.param = ModelParam.parseFrom(paramStream);
            this.dataTransformServer = DataTransformServer.parseFrom(dataTransformServerStream);
            return StatusCode.OK;
        }
        catch (IOException ex){
            LOGGER.error("Protobuffer deserialize error: {}", ex.getMessage());
            return StatusCode.IOERROR;
        }
        catch (Exception ex){
            LOGGER.error("Protobuffer deserialize error: {}", ex.getMessage());
            return StatusCode.UNKNOWNERROR;
        }
    }
}
