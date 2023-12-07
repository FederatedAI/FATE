package org.fedai.osx.broker.queue;

import com.google.protobuf.InvalidProtocolBufferException;
import io.grpc.stub.StreamObserver;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.fedai.osx.broker.constants.MessageFlag;
import org.fedai.osx.core.context.OsxContext;
import org.fedai.osx.core.exceptions.BaseException;
import org.fedai.osx.core.exceptions.ExceptionInfo;
import org.fedai.osx.core.utils.JsonUtil;
import org.ppc.ptp.Osx;

@Slf4j
@Data
public class DirectQueue extends AbstractQueue {

    DataParser inputParser;
    DataParser errorParser = new DataParser() {
        @Override
        public Object parse(Object src) {
            Osx.PushInbound inbound = null;
            try {
                inbound = Osx.PushInbound.parseFrom((byte[]) src);
                ExceptionInfo exceptionInfo = JsonUtil.json2Object(inbound.getPayload().toByteArray(), ExceptionInfo.class);
                return new BaseException(exceptionInfo.getCode(), exceptionInfo.getMessage());
            } catch (InvalidProtocolBufferException e) {
                e.printStackTrace();
            }
            return null;
        }
    };
    StreamObserver streamObserver;

    public DirectQueue(String topic) {
        this.transferId = topic;
    }

    public synchronized void putMessage(OsxContext context, Object data, MessageFlag messageFlag, String msgCode) {
        log.info("putMessage data class {} {}", data.getClass(), data);
        Object newData;
        switch (messageFlag) {
            case SENDMSG:
                newData = inputParser.parse(data);
                streamObserver.onNext(newData);
                break;
            case ERROR:
                newData = errorParser.parse(data);
                streamObserver.onError((Throwable) newData);
                break;
            case COMPELETED:
                streamObserver.onCompleted();
                break;
        }
    }
}
