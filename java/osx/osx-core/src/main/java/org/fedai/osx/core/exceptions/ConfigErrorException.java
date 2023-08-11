package org.fedai.osx.core.exceptions;

import org.fedai.osx.core.constant.StatusCode;

public class ConfigErrorException extends BaseException{

    public ConfigErrorException(String msg){
        super(StatusCode.CONFIG_ERROR,msg);
    }

}
