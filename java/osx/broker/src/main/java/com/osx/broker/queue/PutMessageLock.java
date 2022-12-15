package com.osx.broker.queue;

/**
 * Used when trying to put message
 */
public interface PutMessageLock {
    void lock();

    void unlock();
}
