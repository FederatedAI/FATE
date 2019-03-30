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

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.util.HashMap;
import java.util.Optional;
import java.util.Properties;

import com.webank.ai.fate.core.constant.StatusCode;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

public class Configuration {
    private static final Logger LOGGER = LogManager.getLogger();
    private final String confPath;
    private static HashMap<String, String> properties;

    public Configuration(String confPath){
        this.confPath = confPath;
        properties = new HashMap<>();
    }

    public int load(){
        try{
            Properties pro = new Properties();
            pro.load(new FileInputStream(this.confPath));
            pro.entrySet().forEach(e->{
                Configuration.putProperty((String)e.getKey(), (String)e.getValue());
            });
            return StatusCode.OK;
        }
        catch (FileNotFoundException ex){
            LOGGER.error("Can not found this file: {}", this.confPath);
            return StatusCode.NOFILE;
        }
        catch (Exception ex){
            LOGGER.error(ex);
            return StatusCode.UNKNOWNERROR;
        }
    }

    public static HashMap<String, String> getProperties() {
        return properties;
    }

    public static String getProperty(String key){
        return properties.get(key);
    }

    public static String getProperty(String key, String defaultValue){
        return Optional.ofNullable(properties.get(key)).orElse(defaultValue);
    }

    public static Integer getPropertyInt(String key) {
        if (getProperty(key) == null){
            return null;
        }
        return Integer.parseInt(getProperty(key));
    }

    public static Integer getPropertyInt(String key, int defaultValue) {
        if (getProperty(key) == null){
            return defaultValue;
        }
        return Integer.parseInt(getProperty(key));
    }

    private static void putProperty(String key, String value){
        properties.put(key, value);
    }
}
