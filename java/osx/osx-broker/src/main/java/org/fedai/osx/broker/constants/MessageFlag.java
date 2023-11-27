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
package org.fedai.osx.broker.constants;

public enum MessageFlag {

    SENDMSG(0), ERROR(1), COMPELETED(2), BACKMSG(3);

    private int flag;

    private MessageFlag(int flag) {
        this.flag = flag;
    }

    static public MessageFlag getMessageFlag(int flag) {
        switch (flag) {
            case 0:
                return SENDMSG;
            case 1:
                return ERROR;
            case 2:
                return COMPELETED;
            case 3:
                return BACKMSG;
            default:
                return null;
        }
    }

    public int getFlag() {
        return flag;
    }


}
