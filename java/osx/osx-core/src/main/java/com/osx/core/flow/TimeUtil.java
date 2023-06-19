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

package com.osx.core.flow;

import java.util.concurrent.TimeUnit;

public final class TimeUtil {
    private static volatile long currentTimeMillis = System.currentTimeMillis();

    static {
        Thread daemon = new Thread(new Runnable() {
            @Override
            public void run() {
                while (true) {
                    TimeUtil.currentTimeMillis = System.currentTimeMillis();
                    try {
                        TimeUnit.MILLISECONDS.sleep(1L);
                    } catch (Throwable var2) {
                        ;
                    }
                }
            }
        });
        daemon.setDaemon(true);
        daemon.setName("time-tick-thread");
        daemon.start();
    }

    public TimeUtil() {
    }

    public static long currentTimeMillis() {
        return currentTimeMillis;
    }
}
