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
package org.fedai.osx.core.utils;

import com.google.protobuf.InvalidProtocolBufferException;
import com.google.protobuf.MessageOrBuilder;
import com.google.protobuf.util.JsonFormat;
import org.fedai.osx.core.constant.Dict;

public class ToStringUtils {

    private static JsonFormat.Printer protoPrinter = JsonFormat.printer().preservingProtoFieldNames()

            .includingDefaultValueFields()
            .omittingInsignificantWhitespace();

    public static String toOneLineString(MessageOrBuilder target) {
        if (target != null) {
            try {
                return protoPrinter.print(target);
            } catch (InvalidProtocolBufferException e) {
                return null;
            }
        } else {
            return Dict.NULL_WITH_BRACKETS;
        }
    }


}
