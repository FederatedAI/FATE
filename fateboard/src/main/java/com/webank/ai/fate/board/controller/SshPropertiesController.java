package com.webank.ai.fate.board.controller;

import com.google.common.collect.Maps;
import com.webank.ai.fate.board.global.ErrorCode;
import com.webank.ai.fate.board.global.ResponseResult;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RestController;

import java.io.BufferedInputStream;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.Map;
import java.util.Properties;

@RestController
@RequestMapping(value = "/ssh")

public class SshPropertiesController {
    @Value(value = "${ssh.file.address}")
    String filePath;

    @RequestMapping(value = "/read/{key}", method = RequestMethod.GET)
    public ResponseResult readValue(@PathVariable(value = "key") String key) {
        Properties properties = new Properties();
        HashMap<String, String> data = Maps.newHashMap();
        try {
            BufferedInputStream bufferedInputStream = new BufferedInputStream(new FileInputStream(filePath));
            properties.load(bufferedInputStream);
            String value = properties.getProperty(key);
            data.put(key, value);
            return new ResponseResult<>(ErrorCode.SUCCESS, data);
        } catch (Exception e) {
            e.printStackTrace();
            return new ResponseResult(ErrorCode.PARAM_ERROR);
        }

    }

    @RequestMapping(value = "/remove/{key}", method = RequestMethod.DELETE)
    public ResponseResult removeValue(@PathVariable(value = "key") String key) {
        Properties properties = new Properties();
        try {
            BufferedInputStream bufferedInputStream = new BufferedInputStream(new FileInputStream(filePath));
            properties.load(bufferedInputStream);
            properties.remove(key);
            FileOutputStream fileOutputStream = new FileOutputStream(filePath);

            properties.store(fileOutputStream, "delete '" + key + "' value");
            return new ResponseResult(ErrorCode.SUCCESS);
        } catch (Exception e) {
            e.printStackTrace();
            return new ResponseResult(ErrorCode.PARAM_ERROR);
        }
    }

    @RequestMapping(value = "/read/all", method = RequestMethod.GET)
    public ResponseResult readAll() {
        HashMap<Object, Object> data = Maps.newHashMap();
        Properties properties = new Properties();
        try {
            BufferedInputStream bufferedInputStream = new BufferedInputStream(new FileInputStream(filePath));
            properties.load(bufferedInputStream);
            Enumeration<?> enumeration = properties.propertyNames();
            while (enumeration.hasMoreElements()) {
                String key = (String) enumeration.nextElement();
                String value = properties.getProperty(key);
                data.put(key, value);
            }

        } catch (Exception e) {
            e.printStackTrace();
            return new ResponseResult<>(ErrorCode.PARAM_ERROR);
        }
        return new ResponseResult<Map>(ErrorCode.SUCCESS, data);
    }

    @RequestMapping(value = "add/{key}/{value}", method = RequestMethod.PUT)
    public ResponseResult addProperties(@PathVariable(value = "key") String key, @PathVariable(value = "value") String value) {
        Properties properties = new Properties();
        try {
            FileInputStream fileInputStream = new FileInputStream(filePath);
            properties.load(fileInputStream);
            properties.setProperty(key, value);

            FileOutputStream fileOutputStream = new FileOutputStream(filePath);
            properties.store(fileOutputStream, "add" + "  key:" + key + ", value" + value);
        } catch (Exception e) {
            e.printStackTrace();
            return new ResponseResult<>(ErrorCode.PARAM_ERROR);
        }
        return new ResponseResult<Map>(ErrorCode.SUCCESS);
    }
}
