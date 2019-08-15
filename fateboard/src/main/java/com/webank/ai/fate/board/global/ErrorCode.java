/*
 * Copyright 2019 The FATE Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
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
