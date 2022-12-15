package com.osx.core.token;


public final class TokenResultStatus {

    /**
     * Bad client request.
     */
    public static final int BAD_REQUEST = -4;
    /**
     * Too many request in server.
     */
    public static final int TOO_MANY_REQUEST = -2;
    /**
     * Server or client unexpected failure (due to transport or serialization failure).
     */
    public static final int FAIL = -1;

    /**
     * Token acquired.
     */
    public static final int OK = 0;

    /**
     * Token acquire failed (blocked).
     */
    public static final int BLOCKED = 1;
    /**
     * Should wait for next buckets.
     */
    public static final int SHOULD_WAIT = 2;
    /**
     * Token acquire failed (no rule exists).
     */
    public static final int NO_RULE_EXISTS = 3;
    /**
     * Token acquire failed (reference resource is not available).
     */
    public static final int NO_REF_RULE_EXISTS = 4;
    /**
     * Token acquire failed (strategy not available).
     */
    public static final int NOT_AVAILABLE = 5;
    /**
     * Token is successfully released.
     */
    public static final int RELEASE_OK = 6;
    /**
     * Token already is released before the request arrives.
     */
    public static final int ALREADY_RELEASE=7;

    private TokenResultStatus() {
    }
}
