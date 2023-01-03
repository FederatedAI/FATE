package com.osx.core.exceptions;

import com.osx.core.constant.StatusCode;

public class TransferQueueNotExistException extends BaseException {
    public TransferQueueNotExistException() {
        super(StatusCode.TRANSFER_QUEUE_NOT_FIND, "TRANSFER_QUEUE_NOT_FIND");
    }

    public TransferQueueNotExistException(String code, String msg) {
        super(code, msg);
    }
}
