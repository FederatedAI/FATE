package org.fedai.osx.core.exceptions;

import org.fedai.osx.core.constant.StatusCode;

public class SessionInitException extends BaseException {
    public SessionInitException(String retCode, String message) {
        super(retCode, message);
    }

    public SessionInitException(String message) {
        super(StatusCode.SESSION_INIT_ERROR, message);
    }
}
