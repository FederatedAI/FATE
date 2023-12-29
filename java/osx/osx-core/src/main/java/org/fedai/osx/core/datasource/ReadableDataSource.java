/*
 * Copyright 2019 The FATE Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.fedai.osx.core.datasource;

import org.fedai.osx.core.flow.Property;

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
