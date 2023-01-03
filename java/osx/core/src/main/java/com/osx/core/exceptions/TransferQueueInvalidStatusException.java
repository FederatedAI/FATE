package com.osx.core.exceptions;

import com.osx.core.constant.StatusCode;

public class TransferQueueInvalidStatusException extends BaseException {

    public TransferQueueInvalidStatusException(String msg) {
        super(StatusCode.QUEUE_INVALID_STATUS, msg);
    }


}
