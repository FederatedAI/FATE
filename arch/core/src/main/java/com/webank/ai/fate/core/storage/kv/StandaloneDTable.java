package com.webank.ai.fate.core.storage.kv;

import org.fusesource.lmdbjni.*;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.nio.file.Paths;

public class StandaloneDTable {
    private static final Logger LOGGER = LogManager.getLogger();
    private String dataDir = Paths.get(System.getProperty("user.dir"), "data").toString();
    private Database db;

    public StandaloneDTable(String name, String nameSpace, int partition){
        String path = Paths.get(this.dataDir, "LMDB", nameSpace, name, Integer.toString(partition)).toString();
        Env env = new Env(path);
        this.db = env.openDatabase();
    }

    public byte[] get(String key){
        try{
            return db.get(key.getBytes());
        }
        catch (Exception ex){
            return null;
        }
    }

    public static void main(String[] args){
        StandaloneDTable standaloneDTable = new StandaloneDTable(String.format("%s_%s", "HeteroLRGuest", "2d5374d2471511e9a2e5acde48001122"), "HeteroLR", 0);
        LOGGER.info(standaloneDTable.get("param"));
    }
}
