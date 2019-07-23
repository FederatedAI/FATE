package com.webank.ai.fate.board.global;


public enum ErrorCode {

    //COMMON CODE
    SUCCESS(0, "OK"),
    AUTH_ERROR(10000, "AUTH_ERROR"),
    PARAM_ERROR(10001, "PARAM_ERROR"),
    TIME_OUT(10002, "TIME_OUT"),
    SYSTEM_ERROR(100003, "SYSTEM_ERROR"),
    RUNNING_ERROR(100004, "RUNNING_ERROR"),
    RETURNED_PARAM_ERROR(100005, "ERROR FOR RETURNED PARAMS!"),
    INCOMING_PARAM_ERROR(100006, "ERROR FOR INCOMING PARAMS!");


    private int code;
    private String msg;

    private ErrorCode(int code, String msg) {
        this.code = code;
        this.msg = msg;
    }

    public int getCode() {
        return code;
    }

    public void setCode(int code) {
        this.code = code;
    }

    public String getMsg() {
        return msg;
    }

    public void setMsg(String msg) {
        this.msg = msg;
    }


}
