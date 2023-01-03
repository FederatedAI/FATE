package com.osx.core.exceptions;

import com.osx.core.constant.StatusCode;

public class ConsumeNoMessageException extends BaseException {

    public ConsumeNoMessageException(String code, String msg) {
        super(code, msg);
    }

    public ConsumeNoMessageException(String msg) {
        super(StatusCode.CONSUME_NO_MESSAGE, msg);
    }
}
