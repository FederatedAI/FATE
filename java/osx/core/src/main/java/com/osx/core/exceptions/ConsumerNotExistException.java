package com.osx.core.exceptions;

import com.osx.core.constant.StatusCode;

public class ConsumerNotExistException extends BaseException {

    public ConsumerNotExistException(String msg) {
        super(StatusCode.CONSUMER_NOT_EXIST, msg);
    }

    public ConsumerNotExistException(String code, String msg) {
        super(code, msg);
    }

}
