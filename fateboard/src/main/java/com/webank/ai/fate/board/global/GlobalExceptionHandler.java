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
            response.setMsg(e.getMessage());
        }
        return response;
    }
}
