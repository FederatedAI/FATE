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

import java.util.Map;

public class RollPair {
    public static final String PUT_BATCH = "putBatch";
    public static final String ROLL_PAIR_URI_PREFIX = "v1/roll-pair";
    public static final String EGG_PAIR_URI_PREFIX = "v1/egg-pair";

    public static final String RUN_JOB = "runJob";
    public static final String RUN_TASK = "runTask";
    public static CommandURI EGG_RUN_TASK_COMMAND = new CommandURI(EGG_PAIR_URI_PREFIX + "/" + RUN_TASK);
    public static CommandURI ROLL_RUN_JOB_COMMAND = new CommandURI(ROLL_PAIR_URI_PREFIX + "/" + RUN_JOB);
    ErStore store;
    RollPairContext ctx;
    Map<String, String> options;

    public RollPair(ErStore store, RollPairContext ctx, Map<String, String> options) {
        this.store = store;
        this.ctx = ctx;
        this.options = options;
    }

    public ErStore getStore() {
        return store;
    }

    public void setStore(ErStore store) {
        this.store = store;
    }

    public RollPairContext getCtx() {
        return ctx;
    }

    public void setCtx(RollPairContext ctx) {
        this.ctx = ctx;
    }

    public Map<String, String> getOptions() {
        return options;
    }

    public void setOptions(Map<String, String> options) {
        this.options = options;
    }

//
//    val transferFutures = new Array[Future[Transfer.TransferBatch]](totalPartitions)
//
//
//    public void  putBatch(List<Proxy.Packet> packets, Map<String,String > options){
//        int totalPartitions = store.storeLocator.totalPartitions;
//        Future<ErTask>[]  commandFutures;
//
//        commandFutures = new  Future<ErTask>[totalPartitions];
//        String    tag = options.getOrDefault("job_id_tag", StringConstants.EMPTY);
//        String  jobId = IdUtils.generateJobId( ctx.getSessionId(),tag,"-");
//        options.put("fed_transfer","true");
//        options.put("eggroll.session.id",ctx.getErSession().getSessionId());
//        ErJob job = new ErJob(  jobId,
//                "putAll",
//                Lists.newArrayList(store),
//                Lists.newArrayList(store),
//                Lists.newArrayList(),
//                options);
//        Transfer.TransferBatch.Builder  transferBatchBuilder = Transfer.TransferBatch.newBuilder();
//        Transfer.TransferHeader.Builder transferHeaderBuilder = Transfer.TransferHeader.newBuilder();
//        packets.forEach(packet -> {
//            Transfer.RollSiteHeader rollSiteHeader =null;
//            try {
//                 rollSiteHeader = Transfer.RollSiteHeader.parseFrom(
//                        packet.getHeader().getTask().getModel().getName().getBytes(
//                                StandardCharsets.ISO_8859_1));
//
//
//            } catch (InvalidProtocolBufferException e) {
//                e.printStackTrace();
//            }
//
//            Integer partitionId = Integer.parseInt(rollSiteHeader.getOptionsMap().get("partition_id").toString());
//
//
//        });
//
//
//    }

}


//
//class RollPair(val store: ErStore,
//               val ctx: RollPairContext,
//               val options: Map[String,String] = Map.empty) extends Logging {
//    def putBatch(packets: Iterator[Proxy.Packet],
//                 options: Map[String, String] = Map.empty): Unit = {
//        val totalPartitions = store.storeLocator.totalPartitions
//        val brokers = new Array[FifoBroker[Transfer.TransferBatch]](totalPartitions)
//                val commandFutures = new Array[Future[ErTask]](totalPartitions)
//                val transferFutures = new Array[Future[Transfer.TransferBatch]](totalPartitions)
//                val error = new DistributedRuntimeException()
//
//        val jobId = IdUtils.generateJobId(sessionId = ctx.session.sessionId,
//                tag = options.getOrElse("job_id_tag", StringConstants.EMPTY))
//        val job = ErJob(id = jobId,
//                name = RollPair.PUT_ALL,
//                inputs = Array(store),
//                outputs = Array(store),
//                functors = Array.empty,
//                options = options ++ Map(SessionConfKeys.CONFKEY_SESSION_ID -> ctx.session.sessionId, "fed_transfer" -> "true"))
//
//        val transferBatchBuilder = Transfer.TransferBatch.newBuilder()
//        val transferHeaderBuilder = Transfer.TransferHeader.newBuilder()
//
//        for (packet <- packets) {
//            val rollSiteHeader = RollSiteHeader.parseFrom(
//                    packet.getHeader.getTask.getModel.getName.getBytes(
//                            StandardCharsets.ISO_8859_1)).fromProto()
//
//            val partitionId = rollSiteHeader.options("partition_id").toInt
//
//            val broker = if (transferFutures(partitionId) == null) {
//                val partition = store.partitions(partitionId)
//                val egg = ctx.session.routeToEgg(partition)
//                val task = ErTask(id = IdUtils.generateTaskId(job.id, partitionId, RollPair.PUT_BATCH),
//                        name = RollPair.PUT_ALL,
//                        inputs = Array(partition),
//                        outputs = Array(partition),
//                        job = job)
//
//                val commandFuture = RollPairContext.executor.submit(new Callable[ErTask] {
//                    override def call(): ErTask = {
//                            logTrace(s"thread started for put batch taskId=${task.id}")
//                            val commandClient = new CommandClient(egg.commandEndpoint)
//                            val result = commandClient.call[ErTask](RollPair.EGG_RUN_TASK_COMMAND, task)
//                    logTrace(s"thread ended for put batch taskId=${task.id}")
//                    result
//          }
//                })
//                commandFutures.update(partitionId, commandFuture)
//
//                val newBroker = new FifoBroker[Transfer.TransferBatch]()
//                brokers.update(partitionId, newBroker)
//
//                val internalTransferClient = new InternalTransferClient(egg.transferEndpoint)
//                transferFutures.update(partitionId, internalTransferClient.sendAsync(newBroker))
//
//                newBroker
//            } else {
//                brokers(partitionId)
//            }
//
//            val batch = transferBatchBuilder.setHeader(transferHeaderBuilder.setId(packet.getHeader.getSeq.toInt))
//                    .setData(packet.getBody.getValue)
//                    .build()
//
//            broker.broker.put(batch)
//        }
//
//        brokers.foreach(b => {
//        if (b != null) b.signalWriteFinish()
//    })
//
//        transferFutures.foreach(f => {
//        if (f != null) f.get()
//    })
//
//        commandFutures.foreach(f => {
//        if (f != null) f.get()
//    })
//    }


//}
