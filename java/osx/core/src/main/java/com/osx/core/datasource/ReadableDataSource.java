package com.osx.core.datasource;


import com.osx.core.flow.Property;

public interface ReadableDataSource<S, T> {

    /**
     * Load data data source as the target type.
     *
     * @return the target data.
     * @throws Exception IO or other error occurs
     */
    T loadConfig() throws Exception;

    /**
     * Read original data from the data source.
     *
     * @return the original data.
     * @throws Exception IO or other error occurs
     */
    S readSource() throws Exception;

    /**
     * Get {@link Property} of the data source.
     *
     * @return the property.
     */
    Property<T> getProperty();

    /**
     * Close the data source.
     *
     * @throws Exception IO or other error occurs
     */
    void close() throws Exception;
}
