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
package org.fedai.osx.broker.eggroll;

import com.google.common.cache.CacheBuilder;
import com.google.common.cache.CacheLoader;
import com.google.common.cache.LoadingCache;

import java.util.concurrent.TimeUnit;
public class PutBatchSinkUtil {
    public static LoadingCache<String, ErSession> sessionCache =


            CacheBuilder.newBuilder()
                    .maximumSize(2000)
                    .expireAfterWrite(10, TimeUnit.SECONDS)
                    .concurrencyLevel(100)
                    .recordStats()
                    .softValues()
                    .build(new CacheLoader<String, ErSession>() {
                               @Override
                               public ErSession load(String sessionId) throws Exception {
                                   return new ErSession(sessionId, false);
                               }


                           }
                    );

}
