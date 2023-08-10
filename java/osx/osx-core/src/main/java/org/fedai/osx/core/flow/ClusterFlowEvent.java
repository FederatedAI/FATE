package org.fedai.osx.core.flow;


public enum ClusterFlowEvent {

    /**
     * Normal pass.
     */
    PASS,
    /**
     * Normal block.
     */
    BLOCK,
    /**
     * Token request (from client) passed.
     */
    PASS_REQUEST,
    /**
     * Token request (from client) blocked.
     */
    BLOCK_REQUEST,
    /**
     * Pass (pre-occupy incoming buckets).
     */
    OCCUPIED_PASS,
    /**
     * Block (pre-occupy incoming buckets failed).
     */
    OCCUPIED_BLOCK,
    /**
     * Waiting due to flow shaping or for next bucket tick.
     */
    WAITING
}
