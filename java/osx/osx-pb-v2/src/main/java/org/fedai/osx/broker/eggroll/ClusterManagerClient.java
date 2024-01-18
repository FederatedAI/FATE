/*
 * Copyright 2019 The FATE Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.fedai.osx.broker.eggroll;

import com.google.protobuf.ByteString;
import com.google.protobuf.InvalidProtocolBufferException;
import com.webank.eggroll.core.command.Command;
import com.webank.eggroll.core.meta.Meta;
import org.fedai.osx.core.exceptions.RemoteRpcException;
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
        List<ByteString> result = commandResponse.getResultsList();
        if (result != null) {
            try {
                resultErSessionMeta = ErSessionMeta.parseFromPb(Meta.SessionMeta.parseFrom(result.get(0)));
            } catch (InvalidProtocolBufferException e) {
                new RemoteRpcException("invalid response");
            }
        }
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
                new RemoteRpcException("invalid response");
            }
        }
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
                new RemoteRpcException("invalid response");
            }
        }
        return resultErSessionMeta;
    }

    public ErStore getOrCreateStore(ErStore input) {
        Command.CommandResponse commandResponse = commandClient.call(MetaCommnads.getOrCreateStore, input);
        List<ByteString> result = commandResponse.getResultsList();
        ErStore resultErStore = null;
        if (result != null) {
            try {
                Meta.Store oriStore = Meta.Store.parseFrom(result.get(0));
                resultErStore = ErStore.parseFromPb(oriStore);
            } catch (InvalidProtocolBufferException igore) {

            }
        }
        return resultErStore;
    }

}
