package com.osx.core.exceptions;

import com.osx.core.constant.StatusCode;

public class MessageParseException extends BaseException{
    public MessageParseException(String msg){
        super(StatusCode.MESSAGE_PARSE_ERROR,msg);
    }
}
