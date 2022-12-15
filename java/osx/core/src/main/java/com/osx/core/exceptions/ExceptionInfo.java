package com.osx.core.exceptions;

import com.osx.core.utils.JsonUtil;

public  class ExceptionInfo {


    String code;
        String message;

        public Throwable getThrowable() {
            return throwable;
        }

        public void setThrowable(Throwable throwable) {
            this.throwable = throwable;
        }

        Throwable  throwable;

        public ExceptionInfo() {

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

        public  String  toString(){
            return  JsonUtil.object2Json(this);
        }
    }