
package com.osx.core.flow;

/**
 * @author Eric Zhao
 * @since 1.5.0
 */
public interface OccupySupport {

    /**
     * Try to occupy latter time windows' tokens. If occupy success, a value less than
     * {@code occupyTimeout} in {@link OccupyTimeoutProperty} will be return.
     *
     * <p>
     * Each time we occupy tokens of the future window, current thread should sleep for the
     * corresponding time for smoothing QPS. We can't occupy tokens of the future with unlimited,
     * the sleep time limit is {@code occupyTimeout} in {@link OccupyTimeoutProperty}.
     * </p>
     *
     * @param currentTime  current time millis.
     * @param acquireCount tokens count to acquire.
     * @param threshold    qps threshold.
     * @return time should sleep. Time >= {@code occupyTimeout} in {@link OccupyTimeoutProperty} means
     * occupy fail, in this case, the request should be rejected immediately.
     */
    long tryOccupyNext(long currentTime, int acquireCount, double threshold);

    /**
     * Get current waiting amount. Useful for debug.
     *
     * @return current waiting amount
     */
    long waiting();

    /**
     * Add request that occupied.
     *
     * @param futureTime   future timestamp that the acquireCount should be added on.
     * @param acquireCount tokens count.
     */
    void addWaitingRequest(long futureTime, int acquireCount);

    /**
     * Add occupied pass request, which represents pass requests that borrow the latter windows' token.
     *
     * @param acquireCount tokens count.
     */
    void addOccupiedPass(int acquireCount);

    /**
     * Get current occupied pass QPS.
     *
     * @return current occupied pass QPS
     */
    double occupiedPassQps();
}
