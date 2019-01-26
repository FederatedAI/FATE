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

package com.webank.ai.fate.core.utils;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.util.Properties;

public class PropertyUtils {

    private static final Logger LOGGER = LogManager.getLogger(PropertyUtils.class);
    private static Properties props;

    static {
        loadProps();
    }

    private synchronized static void loadProps() {
        LOGGER.info("Start loading from config.properties...");
        props = new Properties();
        InputStream in = null;
        try {
            in = PropertyUtils.class.getResourceAsStream("/config.properties");
            props.load(in);
        } catch (FileNotFoundException e) {
            LOGGER.error("File not found: config.properties");
        } catch (IOException e) {
            LOGGER.error(e.getMessage());
        } finally {
            try {
                if (null != in) {
                    in.close();
                }
            } catch (IOException e) {
                LOGGER.error("Exception closing config.properties");
            }
        }
        LOGGER.info("Success loading config.properties");
        LOGGER.info("properties：" + props);
    }


    public static String getProperty(String key) {
        return props.getProperty(key);
    }

    public static String getProperty(String key, String defaultValue) {
        return props.getProperty(key, defaultValue);
    }

    public static Long getLongProperty(String key) {
        return Long.parseLong(props.getProperty(key));
    }

    public static Long getLongProperty(String key, Long defaultValue) {
        try {
            return Long.parseLong(props.getProperty(key));
        } catch (Exception e) {
            return defaultValue;
        }
    }
}

