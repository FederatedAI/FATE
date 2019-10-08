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
    SERVLET_ERROR(10000, "Request parameters error !"),
    REQUESTBODY_ERROR(10001,"Requestbody is error "),
    REQUEST_PARAMETER_ERROR(10002,"Some request parameters is null"),
    FATEFLOW_ERROR_CONNECTION(10003, "Can't connect to fateflow module !"),
    FATEFLOW_ERROR_NULL_RESULT(10004, "Results of fateflow are null!"),
    FATEFLOW_ERROR_WRONG_RESULT(10005, "Retcode-result of fateflow is null!"),
    DATABASE_ERROR_RESULT_NULL(10006, "No data in database !"),
    DATA_ERROR (10007, "Error occurs when getting data !"),


    SYSTEM_ERROR(10010, "System error !"),
//    REQUEST_ERROR_BODY(10001, "Requestbody is error !"),
    ERROR_PARAMETER(10008, "Parameters are illegal !"),
//    REQUEST_ERROR_PRARAMETER_MISSING(10004, "Lack of parameter: "),
    DATABASE_ERROR_CONNECTION(10009, "Can't connect to database !"),
    ;


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