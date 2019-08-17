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

package com.webank.ai.fate.core.error.exception;

import com.google.common.collect.Lists;
import org.apache.commons.lang3.exception.ExceptionUtils;

import java.util.List;

public class MultipleRuntimeThrowables extends Throwable {
    private List<Throwable> throwables;

    public MultipleRuntimeThrowables() {
        throwables = Lists.newArrayList();
    }

    public MultipleRuntimeThrowables(String message, List<Throwable> throwables) {
        super(message, new Throwable(stackTraceMessage(throwables)));
    }

    private static String stackTraceMessage(List<Throwable> throwables) {
        StringBuilder sb = new StringBuilder();

        int idx = 0;
        for (Throwable throwable : throwables) {
            sb.append("idx: ")
                    .append(idx++)
                    .append("\n")
                    .append(ExceptionUtils.getStackTrace(throwable))
                    .append("\n\n");
        }

        return sb.toString();
    }
}
