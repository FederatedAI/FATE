package com.webank.ai.fate.common.storage.kv;

import org.fusesource.lmdbjni.*;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

public class StandaloneDTable {
    private static final Logger LOGGER = LogManager.getLogger();
    private String dataDir = "/Users/jarviszeng/Work/Project/FDN/FATE/data/LMDB/";
    private Database db;

    public StandaloneDTable(String name, String nameSpace, int partition){
        String path = String.format("%s/%s/%s/%d", this.dataDir, nameSpace, name, partition);
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
        StandaloneDTable standaloneDTable = new StandaloneDTable(String.format("%s_%s", "HeteroLRGuest", "46490370470711e9984bacde48001122"), "HeteroLR", 0);
        LOGGER.info(standaloneDTable.get("param"));
    }
}
