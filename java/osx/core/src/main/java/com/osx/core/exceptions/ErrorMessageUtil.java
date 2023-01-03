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

package com.osx.core.exceptions;

import com.osx.core.constant.Dict;
import com.osx.core.constant.StatusCode;
import com.osx.core.context.Context;
import io.grpc.Status;
import io.grpc.StatusRuntimeException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashMap;
import java.util.Map;

/**
 * @Description TODO
 * @Author
 **/
public class ErrorMessageUtil {

    static Logger logger = LoggerFactory.getLogger(ErrorMessageUtil.class);

//    public static ReturnResult handleExceptionToReturnResult(Throwable e) {
//        ReturnResult returnResult = new ReturnResult();
//        if (e instanceof BaseException) {
//            BaseException baseException = (BaseException) e;
//            returnResult.setRetcode(baseException.getRetcode());
//            returnResult.setRetmsg(e.getMessage());
//        } else {
//            returnResult.setRetcode(StatusCode.SYSTEM_ERROR);
//        }
//        return returnResult;
//    }

    public static String buildRemoteRpcErrorMsg(int code, String msg) {
        return new StringBuilder().append("host return code ").append(code)
                .append(" host msg :").append(msg).toString();
    }

    public static int transformRemoteErrorCode(int code) {
        return Integer.valueOf(new StringBuilder().append("2").append(code).toString());
    }

    public static String getLocalExceptionCode(Exception e) {
        String retcode = StatusCode.SYSTEM_ERROR;
        if (e instanceof BaseException) {
            retcode = ((BaseException) e).getRetcode();
        }

        return retcode;
    }

    public static StatusRuntimeException throwableToException(Context context, Throwable throwable) {
        if (throwable instanceof StatusRuntimeException) {
            return (StatusRuntimeException) throwable;
        }
        /**
         * 这里组装异常信息
         */
        Status status = Status.fromThrowable(throwable).withDescription("");
        return status.asRuntimeException();
    }


    public static ExceptionInfo handleExceptionExceptionInfo(Context context, Throwable e) {
        ExceptionInfo exceptionInfo = new ExceptionInfo();
        if (e instanceof BaseException) {
            BaseException baseException = (BaseException) e;
            exceptionInfo.setCode(baseException.getRetcode());
            exceptionInfo.setMessage(baseException.getMessage());
        } else {
            exceptionInfo.setCode(StatusCode.SYSTEM_ERROR);
            exceptionInfo.setMessage(e.getMessage());
        }
        exceptionInfo.setThrowable(e);
        if (context.needAssembleException()) {
            exceptionInfo.setThrowable(throwableToException(context, e));
        }
        return exceptionInfo;
    }

    public static Map handleExceptionToMap(Throwable e) {
        Map returnResult = new HashMap();
        if (e instanceof BaseException) {
            BaseException baseException = (BaseException) e;
            returnResult.put(Dict.RET_CODE, baseException.getRetcode());
            returnResult.put(Dict.MESSAGE, baseException.getMessage());
        } else {
            returnResult.put(Dict.RET_CODE, StatusCode.SYSTEM_ERROR);
        }
        return returnResult;
    }

    public static Map handleException(Map result, Throwable e) {
//        if (e instanceof IllegalArgumentException) {
//            result.put(Dict.CODE, StatusCode.PARAM_ERROR);
//            result.put(Dict.MESSAGE, "PARAM_ERROR");
//        } else if (e instanceof NoRouterInfoException) {
//            result.put(Dict.CODE, StatusCode.GUEST_ROUTER_ERROR);
//            result.put(Dict.MESSAGE, "ROUTER_ERROR");
//        } else if (e instanceof SysException) {
//            result.put(Dict.CODE, StatusCode.SYSTEM_ERROR);
//            result.put(Dict.MESSAGE, "SYSTEM_ERROR");
//        } else if (e instanceof OverLoadException) {
//            result.put(Dict.CODE, StatusCode.OVER_LOAD_ERROR);
//            result.put(Dict.MESSAGE, "OVER_LOAD");
//        } else if (e instanceof InvalidRoleInfoException) {
//            result.put(Dict.CODE, StatusCode.INVALID_ROLE_ERROR);
//            result.put(Dict.MESSAGE, "ROLE_ERROR");
//        } else if (e instanceof ShowDownRejectException) {
//            result.put(Dict.CODE, StatusCode.SHUTDOWN_ERROR);
//            result.put(Dict.MESSAGE, "SHUTDOWN_ERROR");
//
//        } else if (e instanceof NoResultException) {
//            logger.error("NET_ERROR ", e);
//            result.put(Dict.CODE, StatusCode.NET_ERROR);
//            result.put(Dict.MESSAGE, "NET_ERROR");
//        } else {
//            logger.error("SYSTEM_ERROR ", e);
//            result.put(Dict.CODE, StatusCode.SYSTEM_ERROR);
//            result.put(Dict.MESSAGE, "SYSTEM_ERROR");
//        }

        return result;
    }
}
