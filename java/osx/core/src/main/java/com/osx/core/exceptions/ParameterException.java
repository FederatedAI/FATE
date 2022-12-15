package com.osx.core.exceptions;


import com.osx.core.constant.StatusCode;

public class ParameterException extends BaseException {

    public ParameterException(String retCode, String message) {
        super(retCode, message);
    }

    public ParameterException(String message) {
        super(StatusCode.PARAM_ERROR, message);
    }

}