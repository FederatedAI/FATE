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

package com.webank.ai.fate.serving.core.utils;

import com.webank.ai.fate.serving.core.bean.EncryptMethod;

import java.security.MessageDigest;

public class EncryptUtils {
    public static String encrypt(String originString, EncryptMethod encryptMethod) {
        try {
            MessageDigest m = MessageDigest.getInstance(getEncryptMethodString(encryptMethod));
            m.update(originString.getBytes("UTF8"));
            byte s[] = m.digest();
            String result = "";
            for (int i = 0; i < s.length; i++) {
                result += Integer.toHexString((0x000000FF & s[i]) | 0xFFFFFF00).substring(6);
            }
            return result;
        } catch (Exception e) {
            e.printStackTrace();
        }

        return "";
    }

    private static String getEncryptMethodString(EncryptMethod encryptMethod){
        String methodString = "";
        switch (encryptMethod){
            case MD5:
                methodString = "MD5";
                break;
            case SHA256:
                methodString = "SHA-256";
                break;
        }
        return methodString;
    }
}
