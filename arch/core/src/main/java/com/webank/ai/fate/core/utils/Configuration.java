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

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;
import java.util.Properties;

import com.webank.ai.fate.core.constant.StatusCode;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.json.JSONObject;

public class Configuration {
    private static final Logger LOGGER = LogManager.getLogger();
    private final String confPath;
    private static String confDirectory;
    private static HashMap<String, String> properties;
    private static HashMap<String, Properties> adapterPropertiesMapPool;
    private static HashMap<String, JSONObject> adapterJsonConfigMapPool;

    static {
        properties = new HashMap<>();
        adapterPropertiesMapPool = new HashMap<>();
        adapterJsonConfigMapPool = new HashMap<>();
    }

    public Configuration(String confPath) {
        this.confPath = confPath;
        confDirectory = Paths.get(confPath).getParent().toString();
    }

    public int load() {
        try {
            Properties pro = new Properties();
            File baseConfFile = new File(this.confPath);
            pro.load(new FileInputStream(baseConfFile));
            pro.entrySet().forEach(e -> {
                Configuration.putProperty((String) e.getKey(), (String) e.getValue());
            });
            loadAdapterConf(baseConfFile.getParent());
            return StatusCode.OK;
        } catch (FileNotFoundException ex) {
            LOGGER.error("Can not found this file: {}", this.confPath);
            return StatusCode.NOFILE;
        } catch (Exception ex) {
            LOGGER.error("", ex);
            return StatusCode.UNKNOWNERROR;
        }
    }

    private void loadAdapterConf(String confRootDir) {
        loadOtherConf(confRootDir, "adapter_conf", adapterPropertiesMapPool, adapterJsonConfigMapPool);
    }

    private void loadOtherConf(String confRootDir, String confDirName,
                               Map<String, Properties> propertiesMapPool,
                               Map<String, JSONObject> jsonMapConfigPool) {
        File loadConfDir = new File(String.format("%s/%s", confRootDir, confDirName));
        if (loadConfDir.exists()) {
            File[] confFiles = loadConfDir.listFiles();
            for (int i = 0; i < confFiles.length; i++) {
                File confFile = confFiles[i];
                try (FileInputStream fileInputStream = new FileInputStream(confFile)) {
                    switch (confFile.getName().split("\\.")[1]) {
                        case "properties":
                            Properties pro = new Properties();
                            pro.load(fileInputStream);
                            propertiesMapPool.put(confFile.getName(), pro);
                            break;
                        case "json":
                            byte[] bytes = new byte[(int) confFile.length()];
                            fileInputStream.read(bytes);
                            JSONObject jsonObject = new JSONObject(new String(bytes,"UTF-8"));
                            jsonMapConfigPool.put(confFile.getName(), jsonObject);
                    }
                } catch (IOException ex) {
                    LOGGER.error(ex);
                }
            }
        }
    }

    public static HashMap<String, String> getProperties() {
        return properties;
    }

    public static Properties getAdapterProperties(String confName) {
        return adapterPropertiesMapPool.get(confName);
    }

    public static JSONObject getAdapterJsonConfig(String confName) {
        return adapterJsonConfigMapPool.get(confName);
    }

    public static String getProperty(String key) {
        return properties.get(key);
    }

    public static String getProperty(String key, String defaultValue) {
        return Optional.ofNullable(properties.get(key)).orElse(defaultValue);
    }

    public static Integer getPropertyInt(String key) {
        if (getProperty(key) == null) {
            return null;
        }
        return Integer.parseInt(getProperty(key));
    }

    public static Integer getPropertyInt(String key, int defaultValue) {
        if (getProperty(key) == null) {
            return defaultValue;
        }
        return Integer.parseInt(getProperty(key));
    }

    private static void putProperty(String key, String value) {
        properties.put(key, value);
    }

    public String getConfPath() {
        return confPath;
    }

    public static String getConfDirectory() {
        return confDirectory;
    }
}
