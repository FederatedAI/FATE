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


import com.osx.api.router.RouterInfo;
import com.osx.broker.constants.Direction;



public class ResourceUtil {

//    static public  String  buildResource(Proxy.Metadata metadata){
//        return "";
//    }

    static public String buildResource(RouterInfo routerInfo, Direction direction) {
        return new StringBuilder().append(routerInfo.getResource()).
                append("-").append(direction.name()).toString();
    }

    static public String buildResource(String resource, Direction direction) {
        return new StringBuilder().append(resource).
                append("-").append(direction.name()).toString();
    }


}
