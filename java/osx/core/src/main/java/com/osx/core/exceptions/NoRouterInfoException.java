package com.osx.core.exceptions;


import com.osx.core.constant.StatusCode;

public class NoRouterInfoException extends BaseException {

    public NoRouterInfoException(String msg) {
        super(StatusCode.PROXY_ROUTER_ERROR, msg);
    }
}
