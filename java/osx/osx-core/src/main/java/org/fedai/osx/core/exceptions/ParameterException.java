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
import org.fedai.osx.core.constant.StatusCode;
public class ParameterException extends BaseException {

    public ParameterException(String retCode, String message) {
        super(retCode, message);
    }

    public ParameterException(String message) {
        super(StatusCode.PARAM_ERROR, message);
    }

}