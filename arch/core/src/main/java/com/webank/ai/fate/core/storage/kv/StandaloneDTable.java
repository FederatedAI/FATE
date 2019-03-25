package com.webank.ai.fate.core.storage.kv;

import org.fusesource.lmdbjni.*;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.nio.file.Paths;

public class StandaloneDTable implements DTable{
    private static final Logger LOGGER = LogManager.getLogger();
    private String dataDir = Paths.get(System.getProperty("user.dir"), "data").toString();
    private Database db;


    @Override
    public void init(String name, String nameSpace, int partition) {
        String path = Paths.get(this.dataDir, "LMDB", nameSpace, name, Integer.toString(partition)).toString();
        LOGGER.info(path);
        Env env = new Env(path);
        this.db = env.openDatabase();
    }


    @Override
    public byte[] get(String key){
        try{
            return db.get(key.getBytes());
        }
        catch (Exception ex){
            return null;
        }
    }
}
