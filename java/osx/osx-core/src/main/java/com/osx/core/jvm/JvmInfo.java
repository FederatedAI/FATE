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


public class JvmInfo {
    long timestamp;
    JVMMemoryUtils.JVMMemoryUsage heap;
    JVMMemoryUtils.JVMMemoryUsage eden;
    JVMMemoryUtils.JVMMemoryUsage old;
    long yongGcCount;
    long yongGcTime;
    long fullGcCount;
    long threadCount;
    long fullGcTime;
    JVMMemoryUtils.JVMMemoryUsage nonHeap;
    JVMMemoryUtils.JVMMemoryUsage survivor;

    public JvmInfo() {
    }

    public JvmInfo(long timestamp) {
        this.timestamp = timestamp;
    }

    @Override
    public String toString() {
        return Long.toString(this.timestamp);
    }

    public long getThreadCount() {
        return threadCount;
    }

    public void setThreadCount(long threadCount) {
        this.threadCount = threadCount;
    }

    public long getYongGcCount() {
        return yongGcCount;
    }

    public void setYongGcCount(long yongGcCount) {
        this.yongGcCount = yongGcCount;
    }

    public long getYongGcTime() {
        return yongGcTime;
    }

    public void setYongGcTime(long yongGcTime) {
        this.yongGcTime = yongGcTime;
    }

    public long getFullGcCount() {
        return fullGcCount;
    }

    public void setFullGcCount(long fullGcCount) {
        this.fullGcCount = fullGcCount;
    }

    public long getFullGcTime() {
        return fullGcTime;
    }

    public void setFullGcTime(long fullGcTime) {
        this.fullGcTime = fullGcTime;
    }

    public long getTimestamp() {
        return timestamp;
    }

    public void setTimestamp(long timestamp) {
        this.timestamp = timestamp;
    }

    public JVMMemoryUtils.JVMMemoryUsage getHeap() {
        return heap;
    }

    public void setHeap(JVMMemoryUtils.JVMMemoryUsage heap) {
        this.heap = heap;
    }

    public JVMMemoryUtils.JVMMemoryUsage getEden() {
        return eden;
    }

    public void setEden(JVMMemoryUtils.JVMMemoryUsage eden) {
        this.eden = eden;
    }

    public JVMMemoryUtils.JVMMemoryUsage getOld() {
        return old;
    }

    public void setOld(JVMMemoryUtils.JVMMemoryUsage old) {
        this.old = old;
    }

    public JVMMemoryUtils.JVMMemoryUsage getNonHeap() {
        return nonHeap;
    }

    public void setNonHeap(JVMMemoryUtils.JVMMemoryUsage nonHeap) {
        this.nonHeap = nonHeap;
    }

    public JVMMemoryUtils.JVMMemoryUsage getSurvivor() {
        return survivor;
    }

    public void setSurvivor(JVMMemoryUtils.JVMMemoryUsage survivor) {
        this.survivor = survivor;
    }

    public void reset() {

    }
}
