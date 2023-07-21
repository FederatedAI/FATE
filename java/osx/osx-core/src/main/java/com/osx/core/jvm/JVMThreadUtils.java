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

package com.osx.core.jvm;

import java.lang.management.ManagementFactory;
import java.lang.management.ThreadMXBean;

/**
 * 类描述：JVM 线程信息工具类
 **/
public class JVMThreadUtils {
    static private ThreadMXBean threadMXBean;

    static {
        threadMXBean = ManagementFactory.getThreadMXBean();
    }

    /**
     * Daemon线程总量
     *
     * @return
     */
    static public int getDaemonThreadCount() {
        return threadMXBean.getDaemonThreadCount();
    }

    /**
     * 当前线程总量
     *
     * @return
     */
    static public int getThreadCount() {
        return threadMXBean.getThreadCount();
    }

    /**
     * 获取线程数量峰值（从启动或resetPeakThreadCount()方法重置开始统计）
     *
     * @return
     */
    static public int getPeakThreadCount() {
        return threadMXBean.getPeakThreadCount();
    }

    /**
     * 获取线程数量峰值（从启动或resetPeakThreadCount()方法重置开始统计），并重置
     *
     * @return
     * @Throws java.lang.SecurityException - if a security manager exists and the caller does not have ManagementPermission("control").
     */
    static public int getAndResetPeakThreadCount() {
        int count = threadMXBean.getPeakThreadCount();
        resetPeakThreadCount();
        return count;
    }

    /**
     * 重置线程数量峰值
     *
     * @Throws java.lang.SecurityException - if a security manager exists and the caller does not have ManagementPermission("control").
     */
    static public void resetPeakThreadCount() {
        threadMXBean.resetPeakThreadCount();
    }

    /**
     * 死锁线程总量
     *
     * @return
     * @Throws IllegalStateException 没有权限或JVM不支持的操作
     */
    static public int getDeadLockedThreadCount() {
        try {
            long[] deadLockedThreadIds = threadMXBean.findDeadlockedThreads();
            if (deadLockedThreadIds == null) {
                return 0;
            }
            return deadLockedThreadIds.length;
        } catch (Exception e) {
            throw new IllegalStateException(e.getMessage(), e);
        }
    }


}

