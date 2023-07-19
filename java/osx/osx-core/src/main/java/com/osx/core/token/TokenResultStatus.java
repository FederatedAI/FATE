package com.osx.core.token;


public final class TokenResultStatus {


    public static final int BAD_REQUEST = -4;

    public static final int TOO_MANY_REQUEST = -2;

    public static final int FAIL = -1;

    public static final int OK = 0;

    public static final int BLOCKED = 1;

    public static final int SHOULD_WAIT = 2;

    public static final int NO_RULE_EXISTS = 3;

    public static final int NO_REF_RULE_EXISTS = 4;

    public static final int NOT_AVAILABLE = 5;

    public static final int RELEASE_OK = 6;

    public static final int ALREADY_RELEASE = 7;

    private TokenResultStatus() {
    }
}
