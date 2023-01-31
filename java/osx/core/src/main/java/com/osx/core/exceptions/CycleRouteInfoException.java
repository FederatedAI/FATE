package com.osx.core.exceptions;

import com.osx.core.constant.StatusCode;

public class CycleRouteInfoException extends BaseException{
    public CycleRouteInfoException(String msg){
        super(StatusCode.CYCLE_ROUTE_ERROR,msg);
    }
}
