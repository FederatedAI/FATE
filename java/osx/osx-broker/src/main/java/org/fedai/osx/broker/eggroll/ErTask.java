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

import com.webank.eggroll.core.meta.Meta;
import org.fedai.osx.core.constant.Dict;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

public class ErTask extends BaseProto<Meta.Task> {

    String id;
    String name = Dict.EMPTY;
    List<ErPartition> inputs = new ArrayList<>();
    List<ErPartition> outputs = new ArrayList<>();
    ErJob job;
    public ErTask(){

    }
    public ErTask(String id, String name, List<ErPartition> inputs, List<ErPartition> outputs, ErJob erJob) {
        this.id = id;
        this.name = name;
        this.inputs = inputs;
        this.outputs = outputs;
        this.job = erJob;
    }

    public static ErTask parseFromPb(Meta.Task task) {
        if (task == null)
            return null;
        String id = task.getId();
        String name = task.getName();
        List<ErPartition> inputs = null;
        if (task.getInputsList() != null) {
            inputs = task.getInputsList().stream().map(ErPartition::parseFromPb).collect(Collectors.toList());
        }
        List<ErPartition> outputs = null;
        if (task.getOutputsList() != null) {
            outputs = task.getOutputsList().stream().map(ErPartition::parseFromPb).collect(Collectors.toList());
        }
        ErJob erJob = ErJob.parseFromPb(task.getJob());
        //String id, String name, List<ErPartition> inputs, List<ErPartition> outputs, ErJob erJob
        ErTask erTask = new ErTask(id, name, inputs, outputs, erJob);
        return erTask;
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

    public List<ErPartition> getInputs() {
        return inputs;
    }

    public void setInputs(List<ErPartition> inputs) {
        this.inputs = inputs;
    }

    public List<ErPartition> getOutputs() {
        return outputs;
    }

    public void setOutputs(List<ErPartition> outputs) {
        this.outputs = outputs;
    }

    public ErJob getJob() {
        return job;
    }

    public void setJob(ErJob job) {
        this.job = job;
    }

    @Override
    Meta.Task toProto() {
        return Meta.Task.newBuilder().setId(this.id).setName(name).addAllInputs(inputs.stream().map(ErPartition::toProto).collect(Collectors.toList()))
                .addAllOutputs(outputs.stream().map(ErPartition::toProto).collect(Collectors.toList())).setJob(job.toProto()).build();

    }


//case class ErTask(id: String,
//                  name: String = StringConstants.EMPTY,
//                  inputs: Array[ErPartition],
//                  outputs: Array[ErPartition],
//                  job: ErJob) extends MetaRpcMessage {
//    def getCommandEndpoint: ErEndpoint = {
//        if (inputs == null || inputs.isEmpty) {
//            throw new IllegalArgumentException("Partition input is empty")
//        }
//
//        val processor = inputs.head.processor
//
//        if (processor == null) {
//            throw new IllegalArgumentException("Head node's input partition is null")
//        }
//
//        processor.commandEndpoint
//    }
//}
}