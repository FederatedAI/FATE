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
package org.fedai.osx.core.exceptions;

import com.google.common.collect.Maps;
import org.fedai.osx.core.constant.Dict;
import org.fedai.osx.core.utils.JsonUtil;

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