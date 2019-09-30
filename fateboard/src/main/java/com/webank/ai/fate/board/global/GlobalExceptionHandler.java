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


import org.apache.http.client.ClientProtocolException;
import org.apache.http.conn.HttpHostConnectException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.converter.HttpMessageNotReadableException;
import org.springframework.web.bind.MissingServletRequestParameterException;
import org.springframework.web.bind.annotation.ControllerAdvice;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.ResponseBody;
import org.springframework.web.bind.annotation.RestController;

import javax.servlet.ServletException;
import javax.servlet.http.HttpServletRequest;
import java.net.SocketException;
import java.sql.SQLIntegrityConstraintViolationException;


@RestController
@ControllerAdvice

public class GlobalExceptionHandler {


    Logger logger = LoggerFactory.getLogger(GlobalExceptionHandler.class);


    @ExceptionHandler(Throwable.class)
    @ResponseBody
    public ResponseResult defaultErrorHandler(HttpServletRequest req, Exception e) throws Exception {


        ResponseResult response = new ResponseResult();

        if (e instanceof ServletException) {
            logger.error("error ", e);
            response.setCode(ErrorCode.SERVLET_ERROR.getCode());
            response.setMsg(ErrorCode.SERVLET_ERROR.getMsg());

        } else if (e instanceof HttpMessageNotReadableException) {
            logger.error("error ", e);
            response.setCode(ErrorCode.REQUESTBODY_ERROR.getCode());
            response.setMsg(ErrorCode.REQUESTBODY_ERROR.getMsg());
        } else if (e instanceof IllegalArgumentException) {
            logger.error("error ", e);
            response.setCode(ErrorCode.ERROR_PARAMETER.getCode());
            response.setMsg(ErrorCode.ERROR_PARAMETER.getMsg());
        } else if (e instanceof SQLIntegrityConstraintViolationException) {
            logger.error("error ", e);
            response.setCode(ErrorCode.DATABASE_ERROR_CONNECTION.getCode());
            response.setMsg(ErrorCode.DATABASE_ERROR_CONNECTION.getMsg());
        } else if (e instanceof SocketException ||e instanceof ClientProtocolException) {
            logger.error("error ", e);
            response.setCode(ErrorCode.FATEFLOW_ERROR_CONNECTION.getCode());
            response.setMsg(ErrorCode.FATEFLOW_ERROR_CONNECTION.getMsg());
        } else {
            logger.error("error ", e);
            response.setCode(ErrorCode.SYSTEM_ERROR.getCode());
            response.setMsg(ErrorCode.SYSTEM_ERROR.getMsg());
        }
        return response;
    }
}