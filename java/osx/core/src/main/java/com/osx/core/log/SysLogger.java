

package com.osx.core.log;

public class SysLogger {

    protected static boolean debugEnabled = false;

    private static boolean quietMode = false;

    private static final String PREFIX = "RocketMQLog: ";
    private static final String ERR_PREFIX = "RocketMQLog:ERROR ";
    private static final String WARN_PREFIX = "RocketMQLog:WARN ";

    public static void setInternalDebugging(boolean enabled) {
        debugEnabled = enabled;
    }

    public static void debug(String msg) {
        if (debugEnabled && !quietMode) {
            System.err.println(PREFIX + msg);
        }
    }

    public static void debug(String msg, Throwable t) {
        if (debugEnabled && !quietMode) {
            System.err.println(PREFIX + msg);
            if (t != null) {
                t.printStackTrace(System.out);
            }
        }
    }

    public static void error(String msg) {
        if (quietMode) {
            return;
        }
        System.err.println(ERR_PREFIX + msg);
    }

    public static void error(String msg, Throwable t) {
        if (quietMode) {
            return;
        }

        System.err.println(ERR_PREFIX + msg);
        if (t != null) {
            t.printStackTrace();
        }
    }

    public static void setQuietMode(boolean quietMode) {
        SysLogger.quietMode = quietMode;
    }

    public static void warn(String msg) {
        if (quietMode) {
            return;
        }

        System.err.println(WARN_PREFIX + msg);
    }

    public static void warn(String msg, Throwable t) {
        if (quietMode) {
            return;
        }

        System.err.println(WARN_PREFIX + msg);
        if (t != null) {
            t.printStackTrace();
        }
    }
}
