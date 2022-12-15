package com.osx.core.exceptions;

import com.osx.core.constant.StatusCode;

public class InvalidRedirectInfoException extends BaseException{
   public InvalidRedirectInfoException(){
        super(StatusCode.INVALID_REDIRECT_INFO,"redirect Info is invalid");
    }
}
