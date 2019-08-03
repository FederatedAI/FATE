package com.webank.ai.fate.board.controller;

import com.google.common.base.Preconditions;
import com.google.common.collect.Maps;
import com.webank.ai.fate.board.global.ErrorCode;
import com.webank.ai.fate.board.global.ResponseResult;
import com.webank.ai.fate.board.utils.Dict;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.InitializingBean;
import org.springframework.core.io.ClassPathResource;
import org.springframework.web.bind.annotation.*;

import java.io.*;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.Map;
import java.util.Properties;

@RestController
@RequestMapping(value = "/ssh")

public class SshPropertiesController {

    private final static Logger logger = LoggerFactory.getLogger(SshPropertiesController.class);


    @RequestMapping(value = "/read/{key}", method = RequestMethod.GET)
    public ResponseResult readValue(@PathVariable(value = "key") String key) throws IOException {
        Properties properties = new Properties();
        HashMap<String, String> data = Maps.newHashMap();

        InputStream inputStream = this.getInputStream();
        properties.load(inputStream);
        String value = properties.getProperty(key);
        data.put(key, value);


        inputStream.close();
        return new ResponseResult<>(ErrorCode.SUCCESS, data);
    }


    @RequestMapping(value = "/remove/{key}", method = RequestMethod.DELETE)
    public ResponseResult removeValue(@PathVariable(value = "key") String key) throws IOException {
        Properties properties = new Properties();
        InputStream inputStream = this.getInputStream();
        properties.load(inputStream);
        properties.remove(key);

        OutputStream writer = this.getOutputStream();
        properties.store(writer, "delete '" + key + "' value");
        writer.close();
        inputStream.close();
        return new ResponseResult(ErrorCode.SUCCESS);
    }


    @RequestMapping(value = "/read/all", method = RequestMethod.GET)
    public ResponseResult readAll() throws IOException {
        HashMap<Object, Object> data = Maps.newHashMap();
        Properties properties = new Properties();
        InputStream inputStream = this.getInputStream();
        properties.load(inputStream);
        Enumeration<?> enumeration = properties.propertyNames();
        while (enumeration.hasMoreElements()) {
            String key = (String) enumeration.nextElement();
            String value = properties.getProperty(key);
            data.put(key, value);
        }

        inputStream.close();
        return new ResponseResult<Map>(ErrorCode.SUCCESS, data);
    }


    @RequestMapping(value = "add/{key}/{value}", method = RequestMethod.PUT)
    public ResponseResult addProperties(@PathVariable(value = "key") String key, @PathVariable(value = "value") String value) throws IOException {
        Properties properties = new Properties();
        InputStream inputStream = this.getInputStream();
        properties.load(inputStream);
        properties.setProperty(key, value);
        OutputStream writer = this.getOutputStream();
        properties.store(writer, "add" + "  key:" + key + ", value" + value);
        return new ResponseResult<Map>(ErrorCode.SUCCESS);
    }


    private InputStream getInputStream() throws IOException {
        String filePath = System.getProperty(Dict.SSH_CONFIG_FILE);
        if (filePath == null || "".equals(filePath)) {
            ClassPathResource classPathResource = new ClassPathResource("ssh.properties");
            return classPathResource.getInputStream();
        } else {
            File file = new File(filePath + "/ssh.properties");
            Preconditions.checkArgument(file.exists() && file.isFile());
            return new BufferedInputStream(new FileInputStream(file));
        }

    }


    private OutputStream getOutputStream() throws FileNotFoundException {
        String filePath = System.getProperty(Dict.SSH_CONFIG_FILE);


        ClassPathResource classPathResource = new ClassPathResource("ssh.properties");
        String path = classPathResource.getPath();
        BufferedOutputStream bufferedOutputStream = new BufferedOutputStream(new FileOutputStream(path));
        return bufferedOutputStream;

    }

}
