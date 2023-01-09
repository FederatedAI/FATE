package com.osx.core.exceptions;

import com.osx.core.constant.StatusCode;

public class ConfigErrorException extends BaseException{

    public ConfigErrorException(String msg){
        super(StatusCode.CONFIG_ERROR,msg);
    }

}
