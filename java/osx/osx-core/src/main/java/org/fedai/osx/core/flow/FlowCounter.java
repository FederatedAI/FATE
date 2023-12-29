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

package org.fedai.osx.core.flow;

import java.util.List;
import java.util.concurrent.atomic.LongAdder;

public class FlowCounter {

    private final LeapArray<LongAdder> data;
    private double qpsAllowed;

    public FlowCounter(double qpsAllowed) {
        this(new UnaryLeapArray(10, 1000), qpsAllowed);
    }

    FlowCounter(LeapArray<LongAdder> data, double qpsAllowed) {
        this.data = data;
        this.qpsAllowed = qpsAllowed;
    }

    public void increment() {
        data.currentWindow().value().increment();
    }

    public void add(int x) {
        data.currentWindow().value().add(x);
    }

    public QpsData getQpsData() {
        long success = 0;
        WindowWrap windowWrap = data.currentWindow();
        List<LongAdder> list = data.values();
        for (LongAdder window : list) {
            success += window.sum();
        }
        double qps = success / data.getIntervalInSecond();

        return new QpsData(windowWrap.windowStart(), getSum());
    }

    public long getSum() {
        data.currentWindow();
        long success = 0;

        List<LongAdder> list = data.values();
        for (LongAdder window : list) {
            success += window.sum();
        }
        return success;
    }

    public double getQps() {
        return getSum() / data.getIntervalInSecond();
    }

    public double getQpsAllowed() {
        return qpsAllowed;
    }

    public FlowCounter setQpsAllowed(double qpsAllowed) {
        this.qpsAllowed = qpsAllowed;
        return this;
    }

    public boolean canPass(int times) {
        return getQps() + times <= qpsAllowed;
    }

    public boolean tryPass(int times) {
        if (canPass(times)) {
            add(times);
            return true;
        }
        return false;
    }

    public class QpsData {
        long current;
        long sum;

        public QpsData(long current, long sum) {
            this.current = current;
            this.sum = sum;
        }
    }
}
