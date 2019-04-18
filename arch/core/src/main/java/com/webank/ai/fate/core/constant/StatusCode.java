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

package com.webank.ai.fate.core.constant;

/**
 * Business logic error > 0
 * System error < 0
 */
public class StatusCode {
    public static final int OK = 0;
    public static final int UNKNOWNERROR = 1;
    public static final int PARAMERROR = 2;
    public static final int ILLEGALDATA = 3;
    public static final int NOMODEL= 4;
    public static final int NOTME = 5;
    public static final int FEDERATEDERROR = 6;
    public static final int TIMEOUT = -1;
    public static final int NOFILE = -2;
    public static final int NETWORKERROR = -3;
    public static final int IOERROR = -4;
    public static final int RUNTIMEERROR = -5;
}
