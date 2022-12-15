package com.osx.core.datasource;


public interface Converter<S, T> {

    /**
     * Convert {@code source} to the target type.
     *
     * @param source the source object
     * @return the target object
     */
    T convert(S source);
}
