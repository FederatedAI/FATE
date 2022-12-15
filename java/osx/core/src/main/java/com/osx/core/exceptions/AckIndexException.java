package com.osx.core.exceptions;

import com.osx.core.constant.StatusCode;

public class AckIndexException extends BaseException{
    public  AckIndexException(){
        super(StatusCode.ACK_INDEX_ERROR,"ACK_INDEX_ERROR");
    }


    public  AckIndexException(String msg){
        super(StatusCode.ACK_INDEX_ERROR,msg);
    }
}
