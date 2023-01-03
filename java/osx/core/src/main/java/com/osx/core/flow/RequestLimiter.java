package com.osx.core.flow;

import com.osx.core.utils.AssertUtil;

import java.util.List;
import java.util.concurrent.atomic.LongAdder;

public class RequestLimiter {

    private final LeapArray<LongAdder> data;
    private double qpsAllowed;

    public RequestLimiter(double qpsAllowed) {
        this(new UnaryLeapArray(10, 1000), qpsAllowed);
    }

    RequestLimiter(LeapArray<LongAdder> data, double qpsAllowed) {
        AssertUtil.isTrue(qpsAllowed >= 0, "max allowed QPS should > 0");
        this.data = data;
        this.qpsAllowed = qpsAllowed;
    }

    public void increment() {
        data.currentWindow().value().increment();
    }

    public void add(int x) {
        data.currentWindow().value().add(x);
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

    public RequestLimiter setQpsAllowed(double qpsAllowed) {
        this.qpsAllowed = qpsAllowed;
        return this;
    }

    public boolean canPass() {
        return getQps() + 1 <= qpsAllowed;
    }

    public boolean tryPass() {
        if (canPass()) {
            add(1);
            return true;
        }
        return false;
    }
}
