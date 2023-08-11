package org.fedai.osx.core.exceptions;

import org.fedai.osx.core.constant.StatusCode;

public class CycleRouteInfoException extends BaseException{
    public CycleRouteInfoException(String msg){
        super(StatusCode.CYCLE_ROUTE_ERROR,msg);
    }
}
