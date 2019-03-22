package com.webank.ai.fate.core.statuscode;

/**
 * Business logic error > 0
 * System error < 0
 */
public class ReturnCode {
    public static int OK = 0;
    public static int UNKNOWNERROR = 1;
    public static int PARAMERROR = 2;
    public static int ILLEGALDATA = 3;
    public static int NOMODEL= 4;
    public static int NOTME = 5;
    public static int TIMEOUT = -1;
    public static int NOFILE = -2;
    public static int IOERROR = -3;
}
