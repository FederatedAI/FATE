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

//public class InferenceRetCode {
//    public static final int OK = 0;
//    public static final int EMPTY_DATA = 100;
//    public static final int NUMERICAL_ERROR = 101;
//    public static final int INVALID_FEATURE = 102;
//    public static final int GET_FEATURE_FAILED = 103;
//    public static final int LOAD_MODEL_FAILED = 104;
//    public static final int NETWORK_ERROR = 105;
//    public static final int DISK_ERROR = 106;
//    public static final int STORAGE_ERROR = 107;
//    public static final int COMPUTE_ERROR = 108;
//    public static final int NO_RESULT = 109;
//    public static final int SYSTEM_ERROR = 110;
//    public static final int ADAPTER_ERROR = 111;
//    public static final int DEAL_FEATURE_FAILED = 112;
//    public static final int NO_FEATURE = 113;
//}


/**
 * guest 参数错误   1100   异常  GuestInvalidParamException
 * host  参数错误   2100   异常  HostInvalidParamException
 * guest 特征错误   1102   异常  GuestInvalidFeatureException
 * host  特征错误   2102   异常  HostInvalidFeatureException
 * host  特征不存在  2113  异常  HostNoFeatureException
 * <p>
 * guest 加载模型失败 1107 异常   GuestLoadModelException
 * host  加载模型失败 1107 异常   HostLoadModelException
 * <p>
 * guest 模型不存在 1104   异常  GuestModelNullException
 * host  模型不存在 2104   异常  HostModelNullException
 * guest 通信异常   1105   异常  GuestNetErrorExcetpion
 * guest 通讯路由不存在  4115 异常  NoRouterInfoException
 * guest  host返回数据异常  1115  HostReturnErrorException
 */

public class BaseException extends RuntimeException {

    protected String retcode;

    public BaseException(String retCode, String message) {
        super(message);
        this.retcode = retCode;
    }

    public BaseException() {

    }

    public String getRetcode() {
        return retcode;
    }
}
