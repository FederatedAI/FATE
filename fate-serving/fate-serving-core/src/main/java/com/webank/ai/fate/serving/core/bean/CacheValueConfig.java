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

package com.webank.ai.fate.serving.core.bean;

import com.google.common.cache.Cache;

public class CacheValueConfig<KT, VT> {
    private int dbIndex;
    private int ttl;
    private Cache<KT, VT> inProcessCache;

    public CacheValueConfig(int dbIndex, int ttl, Cache<KT, VT> inProcessCache) {
        this.dbIndex = dbIndex;
        this.ttl = ttl;
        this.inProcessCache = inProcessCache;
    }

    public int getDbIndex() {
        return dbIndex;
    }

    public int getTtl() {
        return ttl;
    }

    public Cache<KT, VT> getInProcessCache() {
        return inProcessCache;
    }
}
