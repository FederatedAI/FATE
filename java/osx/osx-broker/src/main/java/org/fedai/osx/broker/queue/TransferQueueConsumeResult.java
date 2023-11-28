package org.fedai.osx.broker.queue;

import lombok.Data;
import org.fedai.osx.broker.message.MessageExt;
import org.fedai.osx.broker.message.SelectMappedBufferResult;

@Data
public class TransferQueueConsumeResult {

    SelectMappedBufferResult selectMappedBufferResult;
    long requestIndex;
    long logicIndexTotal;
    String code = "-1";
    MessageExt message;

    public TransferQueueConsumeResult(String code,
                                      SelectMappedBufferResult selectMappedBufferResult,
                                      long requestIndex,
                                      long logicIndex) {
        this.code = code;
        this.selectMappedBufferResult = selectMappedBufferResult;
        this.requestIndex = requestIndex;
        this.logicIndexTotal = logicIndex;
    }
}
