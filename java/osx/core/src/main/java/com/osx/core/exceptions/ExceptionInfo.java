package com.osx.core.exceptions;

import com.google.common.collect.Maps;
import com.osx.core.constant.Dict;
import com.osx.core.utils.JsonUtil;

import java.util.Map;

public class ExceptionInfo {


    String code;
    String message;
    Throwable throwable;

    public ExceptionInfo() {

    }

    public Throwable getThrowable() {
        return throwable;
    }

    public void setThrowable(Throwable throwable) {
        this.throwable = throwable;
    }

    public String getCode() {
        return code;
    }

    public void setCode(String code) {
        this.code = code;
    }

    public String getMessage() {
        return message != null ? message : "";
    }

    public void setMessage(String message) {
        this.message = message;
    }

    public String toString() {
        Map data = Maps.newHashMap();
        data.put(Dict.CODE,code);
        data.put(Dict.MESSAGE,message);
        return JsonUtil.object2Json(data);
    }
}