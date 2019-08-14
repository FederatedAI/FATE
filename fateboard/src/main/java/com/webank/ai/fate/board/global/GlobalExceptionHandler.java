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


import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.web.bind.MissingServletRequestParameterException;
import org.springframework.web.bind.annotation.ControllerAdvice;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.ResponseBody;
import org.springframework.web.bind.annotation.RestController;

import javax.servlet.http.HttpServletRequest;
import java.sql.SQLIntegrityConstraintViolationException;


@RestController
@ControllerAdvice

public class GlobalExceptionHandler {


    Logger logger = LoggerFactory.getLogger(GlobalExceptionHandler.class);


    @ExceptionHandler(Throwable.class)
    @ResponseBody
    public ResponseResult defaultErrorHandler(HttpServletRequest req, Exception e) throws Exception {


        ResponseResult response = new ResponseResult();

        if (e instanceof MissingServletRequestParameterException) {

            response.setCode(ErrorCode.PARAM_ERROR.getCode());
            response.setMsg(e.getMessage());
            logger.error("error ", e);
        } else if (e instanceof IllegalArgumentException) {
            logger.error("error ", e);
            response.setCode(ErrorCode.SYSTEM_ERROR.getCode());
            response.setMsg(e.getMessage());
        } else if (e instanceof SQLIntegrityConstraintViolationException) {
            logger.error("error ", e);
            response.setCode(ErrorCode.SYSTEM_ERROR.getCode());
            response.setMsg(e.getMessage());
        } else {
            logger.error("error ", e);
            response.setCode(ErrorCode.SYSTEM_ERROR.getCode());
            response.setMsg("system error");
        }
        return response;
    }
}
