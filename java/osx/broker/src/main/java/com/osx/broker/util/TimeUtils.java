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
package com.osx.broker.util;

import org.apache.commons.lang3.StringUtils;

import java.text.SimpleDateFormat;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

public class TimeUtils {
    static DateTimeFormatter noSeparatorFormatter = DateTimeFormatter.ofPattern("yyyyMMdd.HHmmss.SSS");

    public static String getNowMs(String dateFormat) {
        LocalDateTime now = LocalDateTime.now();
        if (StringUtils.isBlank(dateFormat)) {
            return noSeparatorFormatter.format(now);
        } else {
            return new SimpleDateFormat(dateFormat).format(now);
        }
    }
}

