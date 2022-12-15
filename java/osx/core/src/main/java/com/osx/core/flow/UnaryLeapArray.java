

package com.osx.core.flow;

import java.util.concurrent.atomic.LongAdder;

public class UnaryLeapArray extends LeapArray<LongAdder> {

    public UnaryLeapArray(int sampleCount, int intervalInMs) {
        super(sampleCount, intervalInMs);
    }

    @Override
    public LongAdder newEmptyBucket(long time) {
        return new LongAdder();
    }

    @Override
    protected WindowWrap<LongAdder> resetWindowTo(WindowWrap<LongAdder> windowWrap, long startTime) {
        windowWrap.resetTo(startTime);
        windowWrap.value().reset();
        return windowWrap;
    }
}
