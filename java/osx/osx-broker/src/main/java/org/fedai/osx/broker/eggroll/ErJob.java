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
package org.fedai.osx.broker.eggroll;

import com.google.common.collect.Lists;
import com.webank.eggroll.core.meta.Meta;
import lombok.Data;

import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

@Data
public class ErJob extends BaseProto<Meta.Job> {


    String id;
    String name;
    List<ErStore> inputs;
    List<ErStore> outputs;
    List<ErJobIO> inputsV2;
    List<ErJobIO> outputsV2;
    List<ErFunctor> functors;
    Map<String, String> options;

    public ErJob(String id, String name
            , List<ErStore> inputsV1,List<ErStore> outputsV1
            , List<ErJobIO> inputsV2,List<ErJobIO> outputsV2
            ,List<ErFunctor> functors
            ,Map<String, String> options) {
        this.id = id;
        this.name = name;
        this.inputs = inputsV1;
        this.outputs = outputsV1;
        this.inputsV2 = inputsV2;
        this.outputsV2 = outputsV2;
        this.functors = functors;
        this.options = options;
    }

    public static ErJob parseFromPb(Meta.Job job) {

        if (job == null)
            return null;
        String id = job.getId();
        String name = job.getName();
        Map<String, String> options = job.getOptionsMap();
        // FATE 1.X  inputMeta
        List<Meta.Store> inputMeta_V1 = job.getInputsList();
        List<ErStore> input_V1 = Lists.newArrayList();
        if (inputMeta_V1 != null) {
            input_V1 = inputMeta_V1.stream().map(ErStore::parseFromPb).collect(Collectors.toList());
        }

        // FATE 2.X  inputMeta
        List<Meta.JobIO> inputMeta_V2 = job.getInputsV2List();
        List<ErJobIO> input_V2 = Lists.newArrayList();
        if (inputMeta_V2 != null) {
            input_V2 = inputMeta_V2.stream().map(ErJobIO::parseFromPb).collect(Collectors.toList());
        }

        // FATE 1.X  outputMeta
        List<Meta.Store> outputMeta_V1 = job.getOutputsList();
        List<ErStore> output_V1 = Lists.newArrayList();
        if (outputMeta_V1 != null) {
            output_V1 = outputMeta_V1.stream().map(ErStore::parseFromPb).collect(Collectors.toList());
        }

        // FATE 2.X  outputMeta
        List<Meta.JobIO> outputMeta_V2 = job.getOutputsV2List();
        List<ErJobIO> output_V2 = Lists.newArrayList();
        if (outputMeta_V2 != null) {
            output_V2 = outputMeta_V2.stream().map(ErJobIO::parseFromPb).collect(Collectors.toList());
        }


        List<ErFunctor> functors = Lists.newArrayList();
        List<Meta.Functor> functorMeta = job.getFunctorsList();
        if (functorMeta != null) {
            functors = functorMeta.stream().map(ErFunctor::parseFromPb).collect(Collectors.toList());
        }

        ErJob erJob = new ErJob(id, name
                ,input_V1,output_V1
                , input_V2,output_V2
                ,functors,
                job.getOptionsMap());
        return erJob;
    }

    public String getId() {
        return id;
    }

    public void setId(String id) {
        this.id = id;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public List<ErJobIO> getInputsV2() {
        return inputsV2;
    }

    public void setInputsV2(List<ErJobIO> inputsV2) {
        this.inputsV2 = inputsV2;
    }

    public List<ErJobIO> getOutputsV2() {
        return outputsV2;
    }

    public void setOutputsV2(List<ErJobIO> outputsV2) {
        this.outputsV2 = outputsV2;
    }

    public List<ErFunctor> getFunctors() {
        return functors;
    }

    public void setFunctors(List<ErFunctor> functors) {
        this.functors = functors;
    }

    public Map<String, String> getOptions() {
        return options;
    }

    public void setOptions(Map<String, String> options) {
        this.options = options;
    }

    @Override
    Meta.Job toProto() {
        return Meta.Job.newBuilder()
                .setId(id)
                .setName(name)
                .addAllFunctors(this.functors.stream().map(ErFunctor::toProto).collect(Collectors.toList()))
                .addAllInputs(inputs.stream().map(ErStore::toProto).collect(Collectors.toList()))
                .addAllOutputs(outputs.stream().map(ErStore::toProto).collect(Collectors.toList()))
                .addAllInputsV2(inputsV2.stream().map(ErJobIO::toProto).collect(Collectors.toList()))
                .addAllOutputsV2(outputsV2.stream().map(ErJobIO::toProto).collect(Collectors.toList()))
                .putAllOptions(options).build();
    }
}