package com.osx.core.exceptions;

import com.osx.core.constant.StatusCode;

public class PutMessageException extends BaseException{

    public PutMessageException(String retCode, String message) {
        super(retCode, message);
    }

    public PutMessageException(String message) {
        super(StatusCode.PUT_MESSAGE_ERROR, message);
    }

}
