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

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public class ErJob extends BaseProto<Meta.Job> {


    String id;
    String name;
    List<ErJobIO> inputs;
    List<ErJobIO> outputs;
    List<ErFunctor> functors;
    Map<String, String> options;

    public ErJob(String id, String name,
                 ErStore input,
                 ErStore output,
                 List<ErFunctor> functors,
                 Map<String, String> options) {
        this.id = id;
        this.name = name;
        this.inputs = new ArrayList<>();
        this.inputs.add(new ErJobIO(input, new ErSerdes(0), new ErSerdes(0), new ErPartitioner(0)));

        this.outputs = new ArrayList<>();
        this.outputs.add(new ErJobIO(output, new ErSerdes(0), new ErSerdes(0), new ErPartitioner(0)));
        this.functors = functors;
        this.options = options;
    }

    public ErJob(String id, String name,
                 List<ErFunctor> functors,
                 Map<String, String> options) {
        this.id = id;
        this.name = name;
        this.inputs = new ArrayList<>();
        this.outputs = new ArrayList<>();
        this.functors = functors;
        this.options = options;
    }

    public static ErJob parseFromPb(Meta.Job job) {

        if (job == null)
            return null;
        String id = job.getId();
        String name = job.getName();
        Map<String, String> options = job.getOptionsMap();
        List<Meta.JobIO> inputMeta = job.getInputsList();
        List<ErJobIO> input = Lists.newArrayList();
        if (inputMeta != null) {
            input = inputMeta.stream().map(ErJobIO::parseFromPb).collect(Collectors.toList());
        }
        List<Meta.JobIO> outputMeta = job.getOutputsList();
        List<ErJobIO> output = Lists.newArrayList();
        if (output != null) {
            output = outputMeta.stream().map(ErJobIO::parseFromPb).collect(Collectors.toList());
        }
        List<ErFunctor> functors = Lists.newArrayList();
        List<Meta.Functor> functorMeta = job.getFunctorsList();
        if (functorMeta != null) {
            functors = functorMeta.stream().map(ErFunctor::parseFromPb).collect(Collectors.toList());
        }

        ErJob erJob = new ErJob(id, name,
                functors,
                job.getOptionsMap());
        erJob.setInputs(input);
        erJob.setOutputs(output);
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

    public List<ErJobIO> getInputs() {
        return inputs;
    }

    public void setInputs(List<ErJobIO> inputs) {
        this.inputs = inputs;
    }

    public List<ErJobIO> getOutputs() {
        return outputs;
    }

    public void setOutputs(List<ErJobIO> outputs) {
        this.outputs = outputs;
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
                .addAllFunctors(this.functors.stream().map(ErFunctor::toProto).collect(Collectors.toList())).
                        addAllInputs(inputs.stream().map(ErJobIO::toProto).collect(Collectors.toList()))
                .addAllOutputs(outputs.stream().map(ErJobIO::toProto).collect(Collectors.toList()))
                .putAllOptions(options).build();
    }
}