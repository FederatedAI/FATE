package com.osx.broker.eggroll;


import com.google.protobuf.InvalidProtocolBufferException;
import com.osx.core.utils.ToStringUtils;
import com.webank.ai.eggroll.api.networking.proxy.Proxy;
import com.webank.eggroll.core.transfer.Transfer;
import io.grpc.stub.StreamObserver;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Future;

public class PutBatchSinkPushRespSO implements StreamObserver<Transfer.TransferBatch> {

    public PutBatchSinkPushRespSO(Proxy.Metadata reqHeader,
                                  Future<ErTask> commandFuture,
                                  StreamObserver<Proxy.Metadata> eggSiteServicerPushRespSO,
                                  CountDownLatch finishLatch
    ) {
        this.reqHeader = reqHeader;
        this.commandFuture = commandFuture;
        this.eggSiteServicerPushRespSO = eggSiteServicerPushRespSO;
        this.finishLatch = finishLatch;
    }

    StreamObserver<Proxy.Metadata> eggSiteServicerPushRespSO;

    Proxy.Metadata reqHeader;

    Future<ErTask> commandFuture;
    CountDownLatch finishLatch;


    Logger logger = LoggerFactory.getLogger(PutBatchSinkPushRespSO.class);

    @Override
    public void onNext(Transfer.TransferBatch resp) {
        //  logTrace(s"PutBatchSinkPushRespSO.onNext calling. rsKey=${rsKey}, rsHeader=${rsHeader}, transferHeader=${oneLineStringTransferHeader}")
        Transfer.TransferHeader transferHeader = resp.getHeader();
        String oneLineStringTransferHeader = ToStringUtils.toOneLineString(transferHeader);

        ErRollSiteHeader rsHeader = null;
        try {
            rsHeader = ErRollSiteHeader.parseFromPb(Transfer.RollSiteHeader.parseFrom(transferHeader.getExt()));
        } catch (InvalidProtocolBufferException e) {
            e.printStackTrace();
        }
        //delim: String = "#", prefix: Array[String] = Array("__rsk")
        String rsKey = rsHeader.getRsKey("#", "__rsk");

        try {
            //logTrace(s"PutBatchSinkPushRespSO.onNext calling, command completing. rsKey=${rsKey}, rsHeader=${rsHeader}, transferHeader=${oneLineStringTransferHeader}")
            commandFuture.get();
            //logTrace(s"PutBatchSinkPushRespSO.onNext calling, command completed. rsKey=${rsKey}, rsHeader=${rsHeader}, transferHeader=${oneLineStringTransferHeader}")
            eggSiteServicerPushRespSO.onNext(reqHeader.toBuilder().setAck(resp.getHeader().getId()).build());
            eggSiteServicerPushRespSO.onCompleted();
        } catch (Exception e) {
            e.printStackTrace();
            logger.error("==========" ,e);

        }

    }

    @Override
    public void onError(Throwable throwable) {
        eggSiteServicerPushRespSO.onError(throwable);
    }

    @Override
    public void onCompleted() {

        finishLatch.countDown();
        //eggSiteServicerPushRespSO.onCompleted();
    }
}


//
//class PutBatchSinkPushRespSO(val reqHeader: Proxy.Metadata,
//                             val commandFuture: Future[ErTask],
//                             val eggSiteServicerPushRespSO_putBatchPollingPushRespSO: StreamObserver[Proxy.Metadata],
//                             val finishLatch: CountDownLatch)
//  extends StreamObserver[Transfer.TransferBatch] with Logging {
//
//  private var transferHeader: Transfer.TransferHeader = _
//  private var oneLineStringTransferHeader: String = _
//
//  private var rsKey: String = _
//  private var rsHeader: ErRollSiteHeader = _
//
//  override def onNext(resp: Transfer.TransferBatch): Unit = {
//    logTrace(s"PutBatchSinkPushRespSO.onNext calling. rsKey=${rsKey}, rsHeader=${rsHeader}, transferHeader=${oneLineStringTransferHeader}")
//    transferHeader = resp.getHeader
//    oneLineStringTransferHeader = ToStringUtils.toOneLineString(transferHeader)
//    rsHeader = RollSiteHeader.parseFrom(transferHeader.getExt).fromProto()
//    rsKey = rsHeader.getRsKey()
//
//    try {
//      logTrace(s"PutBatchSinkPushRespSO.onNext calling, command completing. rsKey=${rsKey}, rsHeader=${rsHeader}, transferHeader=${oneLineStringTransferHeader}")
//      commandFuture.get()
//      logTrace(s"PutBatchSinkPushRespSO.onNext calling, command completed. rsKey=${rsKey}, rsHeader=${rsHeader}, transferHeader=${oneLineStringTransferHeader}")
//      eggSiteServicerPushRespSO_putBatchPollingPushRespSO.onNext(reqHeader.toBuilder.setAck(resp.getHeader.getId).build())
//      eggSiteServicerPushRespSO_putBatchPollingPushRespSO.onCompleted()
//    } catch {
//      case t: Throwable =>
//        onError(t)
//    }
//    logDebug(s"PutBatchSinkPushRespSO.onNext called. rsKey=${rsKey}, rsHeader=${rsHeader}, transferHeader=${oneLineStringTransferHeader}")
//  }
//
//  override def onError(t: Throwable): Unit = {
//    val wrapped = TransferExceptionUtils.throwableToException(t)
//    logError(s"PutBatchSinkPushRespSO.onError calling. rsKey=${rsKey}, rsHeader=${rsHeader}, transferHeader=${oneLineStringTransferHeader}", wrapped)
//    eggSiteServicerPushRespSO_putBatchPollingPushRespSO.onError(wrapped)
//    finishLatch.countDown()
//    logError(s"PutBatchSinkPushRespSO.onError called. rsKey=${rsKey}, rsHeader=${rsHeader}, transferHeader=${oneLineStringTransferHeader}", t)
//  }
//
//  override def onCompleted(): Unit = {
//    logTrace(s"PutBatchSinkPushRespSO.onCompleted calling. rsKey=${rsKey}, rsHeader=${rsHeader}, transferHeader=${oneLineStringTransferHeader}")
//    finishLatch.countDown()
//    logTrace(s"PutBatchSinkPushRespSO.onCompleted called. rsKey=${rsKey}, rsHeader=${rsHeader}, transferHeader=${oneLineStringTransferHeader}")
//  }
//}