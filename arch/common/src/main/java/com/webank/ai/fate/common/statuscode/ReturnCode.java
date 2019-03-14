package com.webank.ai.fate.common.statuscode;

/**
 * Business logic error > 0
 * System error < 0
 */
public class ReturnCode {
    public static int OK = 0;
    public static int PARAMERROR = 1;
    public static int ILLEGALDATA = 2;
    public static int GENERALERROR = 3;
    public static int UNKNOWNERROR = 4;
    public static int TIMEOUT = -1;
    public static int RUNTIMEERROR= -2;
}
