package com.osx.broker.eggroll;

import com.google.protobuf.ByteString;
import com.google.protobuf.InvalidProtocolBufferException;
import com.webank.eggroll.core.command.Command;
import com.webank.eggroll.core.meta.Meta;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;


public class ClusterManagerClient {

    Logger logger = LoggerFactory.getLogger(ClusterManagerClient.class);
    CommandClient commandClient;

    public ClusterManagerClient(CommandClient commandClient) {

        this.commandClient = commandClient;
    }

    public ErSessionMeta getOrCreateSession(ErSessionMeta sessionMeta) {
        if (sessionMeta == null)
            return null;
        ErSessionMeta resultErSessionMeta = null;
        Command.CommandResponse commandResponse = commandClient.call(SessionCommands.getOrCreateSession, sessionMeta);
        logger.info("getOrCreateSession ======================={}", commandResponse);
        List<ByteString> result = commandResponse.getResultsList();
        if (result != null) {
            try {
                resultErSessionMeta = ErSessionMeta.parseFromPb(Meta.SessionMeta.parseFrom(result.get(0)));
            } catch (InvalidProtocolBufferException e) {
                e.printStackTrace();
            }
        }
        logger.info("========= getOrCreateSession param {}  result {}", sessionMeta, resultErSessionMeta);
        return resultErSessionMeta;
    }


    public ErSessionMeta registerSession(ErSessionMeta sessionMeta) {

        Command.CommandResponse commandResponse = commandClient.call(SessionCommands.registerSession, sessionMeta);
        List<ByteString> result = commandResponse.getResultsList();

        ErSessionMeta resultErSessionMeta = null;
        if (result != null) {
            try {
                resultErSessionMeta = ErSessionMeta.parseFromPb(Meta.SessionMeta.parseFrom(result.get(0)));
            } catch (InvalidProtocolBufferException e) {
                e.printStackTrace();
            }

        }
        logger.info("========= registerSession param {} result {}", sessionMeta, resultErSessionMeta);
        return resultErSessionMeta;
    }


    public ErSessionMeta getSession(ErSessionMeta sessionMeta) {
        Command.CommandResponse commandResponse = commandClient.call(SessionCommands.getSession, sessionMeta);
        List<ByteString> result = commandResponse.getResultsList();
        ErSessionMeta resultErSessionMeta = null;
        if (result != null) {
            try {
                resultErSessionMeta = ErSessionMeta.parseFromPb(Meta.SessionMeta.parseFrom(result.get(0)));
            } catch (InvalidProtocolBufferException e) {
                e.printStackTrace();
            }
        }
        logger.info("========= getSession param {}  result {}", sessionMeta, resultErSessionMeta);
        return resultErSessionMeta;

    }

    public ErStore getOrCreateStore(ErStore input) {

        Command.CommandResponse commandResponse = commandClient.call(MetaCommnads.getOrCreateStore, input);
        List<ByteString> result = commandResponse.getResultsList();
        logger.info("==========kkkkkkkkkkkk  {}", commandResponse);
        ErStore resultErStore = null;
        if (result != null) {
            try {
                Meta.Store oriStore = Meta.Store.parseFrom(result.get(0));
                // logger.info("========== oriStore {}",oriStore);
                resultErStore = ErStore.parseFromPb(oriStore);
                //logger.info("=========getOrCreateStore param {} result {}",input,resultErStore);
            } catch (InvalidProtocolBufferException e) {
                e.printStackTrace();
            }
        }
        return resultErStore;
    }

//    def getOrCreateStore(input: ErStoreLocator): ErStore =
//    getOrCreateStore(new ErStore(input, EMPTY_PARTITION_ARRAY, new ConcurrentHashMap[String, String]))
//   def getOrCreateStore(input: ErStore): ErStore = cc.call[ErStore](MetadataCommands.GET_OR_CREATE_STORE, input)


}
