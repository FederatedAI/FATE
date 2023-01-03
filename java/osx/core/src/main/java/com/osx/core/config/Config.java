//package com.osx.core.config;
//
//import org.apache.commons.configuration.ConfigurationException;
//import org.apache.commons.configuration.PropertiesConfiguration;
//
//public class Config {
//
//    PropertiesConfiguration config;
//
//    public Config(String configFile) {
//        initConfig(configFile);
//    }
//
//    public String getString(String property) {
//        String value = config.getString(property);
//        return value;
//    }
//
//    public int getInteger(String property) {
//        Integer value = config.getInt(property);
//        return value;
//    }
//
//    private void initConfig(String configFile) {
//        try {
//            config = new PropertiesConfiguration(configFile);
//        } catch (ConfigurationException e) {
//            e.printStackTrace();
//        }
//    }
//}