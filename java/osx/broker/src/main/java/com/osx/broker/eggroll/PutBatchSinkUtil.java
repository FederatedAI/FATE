package com.osx.broker.eggroll;

import com.google.common.cache.CacheBuilder;
import com.google.common.cache.CacheLoader;
import com.google.common.cache.LoadingCache;

import java.util.concurrent.TimeUnit;

public class PutBatchSinkUtil {
  public static LoadingCache<String ,ErSession> sessionCache =


          CacheBuilder.newBuilder()
                .maximumSize(2000)
                .expireAfterWrite(10, TimeUnit.MINUTES)
                .concurrencyLevel(100)
                .recordStats()
                .softValues()
                .build(new CacheLoader<String, ErSession>() {
                    @Override
                    public ErSession load(String sessionId) throws Exception {
                        return      new ErSession(sessionId , false);
                    }


            }
        );




//    object PutBatchSinkUtils {
//        val sessionCache: LoadingCache[String, ErSession] = CacheBuilder.newBuilder
//                .maximumSize(2000)
//                .expireAfterWrite(10, TimeUnit.MINUTES)
//                .concurrencyLevel(100)
//                .recordStats
//                .softValues
//                .build(new CacheLoader[String, ErSession]() {
//            override def load(key: String): ErSession = {
//                    new ErSession(sessionId = key, createIfNotExists = false)
//            }
//        })
//    }
}
