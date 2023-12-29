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

package org.fedai.osx.broker.zk;

import org.apache.zookeeper.KeeperException;

import java.util.List;
import java.util.concurrent.Executor;

public interface ZookeeperClient {

    void create(String path, boolean ephemeral);

    void delete(String path);

    List<String> getChildren(String path);

    List<String> addChildListener(String path, ChildListener listener);

    void addDataListener(String path, DataListener listener);

    void addDataListener(String path, DataListener listener, Executor executor);

    void removeDataListener(String path, DataListener listener);

    void removeChildListener(String path, ChildListener listener);

    void addStateListener(StateListener listener);

    void removeStateListener(StateListener listener);

    boolean isConnected();

    void close();

    void create(String path, String content, boolean ephemeral) throws KeeperException.NodeExistsException;

    String getContent(String path);

    void clearAcl(String path);

    boolean checkExists(String path);

}
