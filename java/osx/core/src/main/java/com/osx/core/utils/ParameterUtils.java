package com.osx.core.utils;

import com.osx.core.exceptions.ParameterException;
import org.checkerframework.checker.nullness.qual.Nullable;


public class ParameterUtils {
    public static void checkArgument(boolean expression, @Nullable String errorMessage) {
        if (!expression) {
            throw new ParameterException(errorMessage);
        }
    }
}
