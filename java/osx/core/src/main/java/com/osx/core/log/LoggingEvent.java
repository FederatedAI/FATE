

package com.osx.core.log;

import java.io.*;
import java.util.ArrayList;

public class LoggingEvent implements Serializable {

    transient public final String fqnOfCategoryClass;

    transient private Object message;

    transient private Level level;

    transient private Logger logger;

    private String renderedMessage;

    private String threadName;

    public final long timeStamp;

    private Throwable throwable;

    public LoggingEvent(String fqnOfCategoryClass, Logger logger,
                        Level level, Object message, Throwable throwable) {
        this.fqnOfCategoryClass = fqnOfCategoryClass;
        this.message = message;
        this.logger = logger;
        this.throwable = throwable;
        this.level = level;
        timeStamp = System.currentTimeMillis();
    }

    public Object getMessage() {
        if (message != null) {
            return message;
        } else {
            return getRenderedMessage();
        }
    }

    public String getRenderedMessage() {
        if (renderedMessage == null && message != null) {
            if (message instanceof String) {
                renderedMessage = (String) message;
            } else {
                renderedMessage = message.toString();
            }
            if (renderedMessage != null) {
                renderedMessage = renderedMessage.replace('\r', ' ').replace('\n', ' ');
            }
        }
        return renderedMessage;
    }

    public String getThreadName() {
        if (threadName == null) {
            threadName = (Thread.currentThread()).getName();
        }
        return threadName;
    }

    public Level getLevel() {
        return level;
    }

    public String getLoggerName() {
        return logger.getName();
    }

    public String[] getThrowableStr() {
        if (throwable == null) {
            return null;
        }
        StringWriter sw = new StringWriter();
        PrintWriter pw = new PrintWriter(sw);
        try {
            throwable.printStackTrace(pw);
        } catch (RuntimeException ex) {
            SysLogger.warn("InnerLogger print stack trace error", ex);
        }
        pw.flush();
        LineNumberReader reader = new LineNumberReader(
            new StringReader(sw.toString()));
        ArrayList<String> lines = new ArrayList<String>();
        try {
            String line = reader.readLine();
            while (line != null) {
                lines.add(line);
                line = reader.readLine();
            }
        } catch (IOException ex) {
            if (ex instanceof InterruptedIOException) {
                Thread.currentThread().interrupt();
            }
            lines.add(ex.toString());
        }
        String[] tempRep = new String[lines.size()];
        lines.toArray(tempRep);
        return tempRep;
    }
}
