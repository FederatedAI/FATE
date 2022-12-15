
package com.osx.core.flow;

/**
 * Function functional interface from JDK 8.
 */
public interface Function<T, R> {

    /**
     * Applies this function to the given argument.
     *
     * @param t the function argument
     * @return the function result
     */
    R apply(T t);
}
