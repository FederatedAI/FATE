package org.fedai.osx.core.exceptions;

import org.fedai.osx.core.constant.StatusCode;

public class InvalidRequestException extends BaseException {
    public InvalidRequestException(String msg){
        super(StatusCode.PTP_INVALID_REQUEST,msg);
    }
}
