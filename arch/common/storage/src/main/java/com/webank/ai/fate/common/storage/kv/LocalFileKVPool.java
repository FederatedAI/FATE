package com.webank.ai.fate.common.storage.kv;

import org.apache.commons.lang3.StringUtils;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Map;

public class LocalFileKVPool extends BaseKVPool<String, byte[]>{
    protected String dir = "/tmp/jarvis/test";
    public LocalFileKVPool(){
        Path dirPath = Paths.get(this.dir);
        try{
            if(!Files.exists(dirPath)){
                Files.createDirectories(dirPath);
            }
            else if (!Files.isDirectory(dirPath)){
                Files.delete(dirPath);
                Files.createDirectories(dirPath);
            }

        }
        catch (FileNotFoundException ex){
            System.out.println("not found");
        }
        catch (IOException ex){
            System.out.println("xxxx");
        }
    }

    @Override
    public byte[] put(String key, byte[] value){
        try{
            Files.write(Paths.get(this.dir, key), value);
        }
        catch (FileNotFoundException ex){
            System.out.println("not file");
        }
        catch (Exception ex){
            System.out.println("not file");
        }
        return null;
    }

    @Override
    public byte[] putIfAbsent(String key, byte[] value){
        if (!Files.exists(Paths.get(this.dir, key))){
            return this.put(key, value);
        }
        else{
            return null;
        }
    }

    @Override
    public void putAll(Map<String, byte[]> kv){
        for (Map.Entry<String, byte[]> entry : kv.entrySet()){
            this.put(entry.getKey(), entry.getValue());
        }
    }

    @Override
    public byte[] get(String key){
        if (Files.exists(Paths.get(this.dir, key))){
            try{
                return Files.readAllBytes(Paths.get(this.dir, key));
            }
            catch (Exception ex){
                System.out.println("test");
                return null;
            }
        }
        else{
            return null;
        }
    }

    public String joinKeys(String[] keys){
        return StringUtils.join(keys, "-");
    }
}
