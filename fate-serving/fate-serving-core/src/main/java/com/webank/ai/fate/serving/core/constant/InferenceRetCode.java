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

package com.webank.ai.fate.serving.core.constant;

public class InferenceRetCode {
    public static final int OK = 0;
    public static final int EMPTY_DATA = 100;
    public static final int NUMERICAL_ERROR = 101;
    public static final int INVALID_FEATURE = 102;
    public static final int GET_FEATURE_FAILED = 103;
    public static final int LOAD_MODEL_FAILED = 104;
    public static final int NETWORK_ERROR = 105;
    public static final int DISK_ERROR = 106;
    public static final int STORAGE_ERROR = 107;
    public static final int COMPUTE_ERROR = 108;
    public static final int NO_RESULT = 109;
    public static final int SYSTEM_ERROR = 110;
    public static final int ADAPTER_ERROR = 111;
    public static final int DEAL_FEATURE_FAILED = 112;
    public static final int NO_FEATURE = 113;
}
