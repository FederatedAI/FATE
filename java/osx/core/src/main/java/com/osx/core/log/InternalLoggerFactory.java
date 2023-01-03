package com.osx.core.log;

import java.util.concurrent.ConcurrentHashMap;

public abstract class InternalLoggerFactory {

    public static final String LOGGER_SLF4J = "slf4j";

    public static final String LOGGER_INNER = "inner";

    public static final String DEFAULT_LOGGER = LOGGER_SLF4J;

    private static String loggerType = null;

    private static ConcurrentHashMap<String, InternalLoggerFactory> loggerFactoryCache = new ConcurrentHashMap<String, InternalLoggerFactory>();

    static {
        try {
            new Slf4jLoggerFactory();
        } catch (Throwable e) {
            //ignore
        }
        try {
            new InnerLoggerFactory();
        } catch (Throwable e) {
            //ignore
        }
    }

    public static InternalLogger getLogger(Class clazz) {
        return getLogger(clazz.getName());
    }

    public static InternalLogger getLogger(String name) {
        return getLoggerFactory().getLoggerInstance(name);
    }

    private static InternalLoggerFactory getLoggerFactory() {
        InternalLoggerFactory internalLoggerFactory = null;
        if (loggerType != null) {
            internalLoggerFactory = loggerFactoryCache.get(loggerType);
        }
        if (internalLoggerFactory == null) {
            internalLoggerFactory = loggerFactoryCache.get(DEFAULT_LOGGER);
        }
        if (internalLoggerFactory == null) {
            internalLoggerFactory = loggerFactoryCache.get(LOGGER_INNER);
        }
        if (internalLoggerFactory == null) {
            throw new RuntimeException("[RocketMQ] Logger init failed, please check logger");
        }
        return internalLoggerFactory;
    }

    public static void setCurrentLoggerType(String type) {
        loggerType = type;
    }

    protected void doRegister() {
        String loggerType = getLoggerType();
        if (loggerFactoryCache.get(loggerType) != null) {
            return;
        }
        loggerFactoryCache.put(loggerType, this);
    }

    protected abstract void shutdown();

    protected abstract InternalLogger getLoggerInstance(String name);

    protected abstract String getLoggerType();
}
