package org.fedai.osx.core.utils;
 
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.util.Properties;

 

public final class PropertiesUtil
{
    public static Logger logger = LoggerFactory.getLogger(PropertiesUtil.class);
 
    public static Properties getProperties(String path)
    {
        Properties prop = new Properties();

        loadProp(prop, path);
 
        return prop;
    }
 
    private static void loadProp(Properties p, String conf)
    {
        InputStream is = getInputStream(conf);
 
        if(null != is)
        {
            try
            {
                p.load(is);
            }
            catch (IOException e)
            {
                logger.info("file not found!");
            }
            finally
            {
                if(is != null)
                {
                    try
                    {
                        is.close();
                    }
                    catch (IOException e)
                    {
                        logger.info("stream close fail!");
                    }
                 }
            }
        }
    }
 
    //获取输入流
    private static InputStream getInputStream(String conf)
    {
        File file = new File(conf);
        InputStream is = null;
        try {
            is = new BufferedInputStream(new FileInputStream(file));
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        return is;
    }
 
    //获取输出流
    private static OutputStream getOutPutStream(String conf)
    {
        ClassLoader classLoader = Thread.currentThread().getContextClassLoader();
 
        OutputStream out = null;
 
        if(null != classLoader)
        {
            String filePath = classLoader.getResource(conf).getFile();
            try
            {
                out = new FileOutputStream(filePath);
            }
            catch (FileNotFoundException e)
            {
                logger.info("file not found!!!");
            }
        }
        return out;
    }
 
    //根据key读取value
    public static String getValue(Properties p, String key)
    {
        String value = p.getProperty(key);
 
        return value == null?"":value;
    }
 
    //设置key=value
    public static void setValue(String conf, String key, String value)
    {
        Properties p = getProperties(conf);
 
        OutputStream out = getOutPutStream(conf);
 
        p.setProperty(key, value);
 
        try
        {
             p.store(out, "set:"+key+"="+value);
        }
        catch (IOException e)
        {
            logger.info("set properties fail!!!");
        }
        finally
        {
            if(out != null)
            {
                try
                {
                    out.close();
                }
                catch (IOException e)
                {
                    logger.info("stream close fail!");
                }
            }
        }
    }
 
}