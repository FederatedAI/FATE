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
package com.osx.broker.constants;

public enum MessageFlag {

    MSG(0), ERROR(1), COMPELETED(2);

    private int flag;

    private MessageFlag(int flag) {
        this.flag = flag;
    }

    static public MessageFlag getMessageFlag(int flag) {
        switch (flag) {
            case 0:
                return MSG;
            case 1:
                return ERROR;
            case 2:
                return COMPELETED;
            default:
                return null;
        }
    }

    public int getFlag() {
        return flag;
    }


}
