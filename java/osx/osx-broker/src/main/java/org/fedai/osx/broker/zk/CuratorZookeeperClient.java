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


import org.apache.commons.lang3.StringUtils;
import org.apache.curator.framework.CuratorFramework;
import org.apache.curator.framework.CuratorFrameworkFactory;
import org.apache.curator.framework.api.CuratorWatcher;
import org.apache.curator.framework.recipes.cache.TreeCache;
import org.apache.curator.framework.recipes.cache.TreeCacheEvent;
import org.apache.curator.framework.recipes.cache.TreeCacheListener;
import org.apache.curator.framework.state.ConnectionState;
import org.apache.curator.framework.state.ConnectionStateListener;
import org.apache.curator.retry.RetryNTimes;
import org.apache.zookeeper.CreateMode;
import org.apache.zookeeper.KeeperException.NoNodeException;
import org.apache.zookeeper.KeeperException.NodeExistsException;
import org.apache.zookeeper.WatchedEvent;
import org.apache.zookeeper.ZooDefs;
import org.apache.zookeeper.data.ACL;
import org.apache.zookeeper.data.Id;
import org.apache.zookeeper.server.auth.DigestAuthenticationProvider;
import org.fedai.osx.core.config.MetaInfo;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executor;


public class CuratorZookeeperClient extends AbstractZookeeperClient<CuratorZookeeperClient.CuratorWatcherImpl, CuratorZookeeperClient.CuratorWatcherImpl> {

    static final Charset CHARSET = Charset.forName("UTF-8");
    private static final Logger logger = LoggerFactory.getLogger(CuratorZookeeperClient.class);
    private static final String SCHEME = "digest";
    private final CuratorFramework client;
    private Map<String, TreeCache> treeCacheMap = new ConcurrentHashMap<>();
    private boolean aclEnable;
    private String aclUsername;
    private String aclPassword;
    private List<ACL> acls = new ArrayList<>();

    public CuratorZookeeperClient(ZkConfig zkConfig) {
        super(zkConfig);
        try {
            int timeout = zkConfig.getTimeout();
            CuratorFrameworkFactory.Builder builder = CuratorFrameworkFactory.builder()
                    .connectString(zkConfig.getAddress())
                    .retryPolicy(new RetryNTimes(1, 1000))
                    .connectionTimeoutMs(timeout);

            aclEnable = MetaInfo.PROPERTY_ACL_ENABLE;
            if (aclEnable) {
                aclUsername = MetaInfo.PROPERTY_ACL_USERNAME;
                aclPassword = MetaInfo.PROPERTY_ACL_PASSWORD;

                if (StringUtils.isBlank(aclUsername) || StringUtils.isBlank(aclPassword)) {
                    aclEnable = false;
                    MetaInfo.PROPERTY_ACL_ENABLE = false;
                } else {
                    builder.authorization(SCHEME, (aclUsername + ":" + aclPassword).getBytes());

                    Id allow = new Id(SCHEME, DigestAuthenticationProvider.generateDigest(aclUsername + ":" + aclPassword));
                    // add more
                    acls.add(new ACL(ZooDefs.Perms.ALL, allow));
                }
            }

            client = builder.build();
            client.getConnectionStateListenable().addListener(new ConnectionStateListener() {
                @Override
                public void stateChanged(CuratorFramework client, ConnectionState state) {

                    if (state == ConnectionState.LOST) {
                        CuratorZookeeperClient.this.stateChanged(StateListener.DISCONNECTED);
                    } else if (state == ConnectionState.CONNECTED) {
                        CuratorZookeeperClient.this.stateChanged(StateListener.CONNECTED);


                    } else if (state == ConnectionState.RECONNECTED) {
                        CuratorZookeeperClient.this.stateChanged(StateListener.RECONNECTED);
                    }
                }
            });
            client.start();

            if (aclEnable) {
                client.setACL().withACL(acls).forPath("/");
            }
        } catch (Exception e) {
            throw new IllegalStateException(e.getMessage(), e);
        }
    }

    @Override
    public void createPersistent(String path) {
        try {
            if (logger.isDebugEnabled()) {
                logger.debug("createPersistent {}", path);
            }
            if (aclEnable) {
                client.create().withACL(acls).forPath(path);
            } else {
                client.create().forPath(path);
            }
        } catch (NodeExistsException e) {
        } catch (Exception e) {
            throw new IllegalStateException(e.getMessage(), e);
        }
    }

    @Override
    public void createEphemeral(String path) {
        try {
            if (logger.isDebugEnabled()) {
                logger.debug("createEphemeral {}", path);
            }
            if (aclEnable) {
                client.create().withMode(CreateMode.EPHEMERAL).withACL(acls).forPath(path);
            } else {
                client.create().withMode(CreateMode.EPHEMERAL).forPath(path);
            }
        } catch (NodeExistsException e) {
        } catch (Exception e) {
            throw new IllegalStateException(e.getMessage(), e);
        }
    }

    @Override
    protected void createPersistent(String path, String data) throws NodeExistsException {
        byte[] dataBytes = data.getBytes(CHARSET);
        try {
            if (logger.isDebugEnabled()) {
                logger.debug("createPersistent {} data {}", path, data);
            }
            if (aclEnable) {
                client.create().withACL(acls).forPath(path, dataBytes);
            } else {
                client.create().forPath(path, dataBytes);
            }
        } catch (NodeExistsException e) {
            throw e;
        } catch (Exception e) {
            throw new IllegalStateException(e.getMessage(), e);
        }
    }

    @Override
    public void createEphemeral(String path, String data) throws NodeExistsException {
        byte[] dataBytes = data.getBytes(CHARSET);
        try {
            if (logger.isDebugEnabled()) {
                logger.debug("createEphemeral {} data {}", path, data);
            }
            if (aclEnable) {
                client.create().withMode(CreateMode.EPHEMERAL).withACL(acls).forPath(path, dataBytes);
            } else {
                client.create().withMode(CreateMode.EPHEMERAL).forPath(path, dataBytes);
            }
        } catch (NodeExistsException e) {

            throw e;
//            try {
////                if (aclEnable) {
////                    Stat stat = client.checkExists().forPath(path);
////                    client.setData().withVersion(stat.getAversion()).forPath(path, dataBytes);
////                } else {
////                    client.setData().forPath(path, dataBytes);
////                }
//            } catch (Exception e1) {
//                throw new IllegalStateException(e.getMessage(), e1);
//            }
        } catch (Exception e) {
            throw new IllegalStateException(e.getMessage(), e);
        }
    }

    @Override
    public void delete(String path) {
        try {
            if (aclEnable) {
                this.clearAcl(path);
            }
            client.delete().forPath(path);
        } catch (NoNodeException e) {
            e.printStackTrace();
        } catch (Exception e) {
            e.printStackTrace();
            throw new IllegalStateException(e.getMessage(), e);
        } catch (Throwable e) {
            e.printStackTrace();
        }
    }

    public void close() {
        client.close();
    }

    @Override
    public List<String> getChildren(String path) {
        try {
            return client.getChildren().forPath(path);
        } catch (NoNodeException e) {
            return null;
        } catch (Exception e) {
            throw new IllegalStateException(e.getMessage(), e);
        }
    }

    @Override
    public boolean checkExists(String path) {
        try {
            if (client.checkExists().forPath(path) != null) {
                return true;
            }
        } catch (Exception e) {
        }
        return false;
    }

    @Override
    public boolean isConnected() {
        return client.getZookeeperClient().isConnected();
    }

    @Override
    public String doGetContent(String path) {
        try {
            byte[] dataBytes = client.getData().forPath(path);
            return (dataBytes == null || dataBytes.length == 0) ? null : new String(dataBytes, CHARSET);
        } catch (NoNodeException e) {
            // ignore NoNode Exception.
        } catch (Exception e) {
            throw new IllegalStateException(e.getMessage(), e);
        }
        return null;
    }

    @Override
    public void doClose() {
        if (aclEnable) {
            this.clearAcl("/");
        }
        client.close();
    }

    @Override
    public CuratorWatcherImpl createTargetChildListener(String path, ChildListener listener) {
        return new CuratorWatcherImpl(client, listener);
    }

    @Override
    public List<String> addTargetChildListener(String path, CuratorWatcherImpl listener) {
        try {
            return client.getChildren().usingWatcher(listener).forPath(path);
        } catch (NoNodeException e) {
            return null;
        } catch (Exception e) {
            throw new IllegalStateException(e.getMessage(), e);
        }
    }

    @Override
    protected CuratorWatcherImpl createTargetDataListener(String path, DataListener listener) {
        return new CuratorWatcherImpl(client, listener);
    }

    @Override
    protected void addTargetDataListener(String path, CuratorWatcherImpl treeCacheListener) {
        this.addTargetDataListener(path, treeCacheListener, null);
    }

    @Override
    protected void addTargetDataListener(String path, CuratorWatcherImpl treeCacheListener, Executor executor) {
        try {
            TreeCache treeCache = TreeCache.newBuilder(client, path).setCacheData(false).build();
            treeCacheMap.putIfAbsent(path, treeCache);

            if (executor == null) {
                treeCache.getListenable().addListener(treeCacheListener);
            } else {
                treeCache.getListenable().addListener(treeCacheListener, executor);
            }

            treeCache.start();
        } catch (Exception e) {
            throw new IllegalStateException("Add treeCache listener for path:" + path, e);
        }
    }

    @Override
    protected void removeTargetDataListener(String path, CuratorWatcherImpl treeCacheListener) {
        TreeCache treeCache = treeCacheMap.get(path);
        if (treeCache != null) {
            treeCache.getListenable().removeListener(treeCacheListener);
        }
        treeCacheListener.dataListener = null;
    }

    @Override
    public void removeTargetChildListener(String path, CuratorWatcherImpl listener) {
        listener.unwatch();
    }

    @Override
    public void clearAcl(String path) {
        if (aclEnable) {
            if (logger.isDebugEnabled()) {
                logger.debug("clear acl {}", path);
            }
            try {
                client.setACL().withACL(ZooDefs.Ids.OPEN_ACL_UNSAFE).forPath(path);
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }

    /**
     * just for unit test
     *
     * @return
     */
    CuratorFramework getClient() {
        return client;
    }

    public static class CuratorWatcherImpl implements CuratorWatcher, TreeCacheListener {

        private CuratorFramework client;
        private volatile ChildListener childListener;
        private volatile DataListener dataListener;


        public CuratorWatcherImpl(CuratorFramework client, ChildListener listener) {
            this.client = client;
            this.childListener = listener;
        }

        public CuratorWatcherImpl(CuratorFramework client, DataListener dataListener) {
            this.dataListener = dataListener;
        }

        protected CuratorWatcherImpl() {
        }

        public void unwatch() {
            this.childListener = null;
        }

        @Override
        public void process(WatchedEvent event) throws Exception {

            if (childListener != null) {
                String path = event.getPath() == null ? "" : event.getPath();
                childListener.childChanged(path,
                        // if path is null, curator using watcher will throw NullPointerException.
                        // if client connect or disconnect to server, zookeeper will queue
                        // watched event(Watcher.Event.EventType.None, .., path = null).
                        StringUtils.isNotEmpty(path)
                                ? client.getChildren().usingWatcher(this).forPath(path)
                                : Collections.<String>emptyList());
            }

        }

        @Override
        public void childEvent(CuratorFramework client, TreeCacheEvent event) throws Exception {
            if (dataListener != null) {
                if (logger.isDebugEnabled()) {
                    logger.debug("listen the zookeeper changed. The changed data:" + event.getData());
                }
                TreeCacheEvent.Type type = event.getType();
                EventType eventType = null;
                String content = null;
                String path = null;
                switch (type) {
                    case NODE_ADDED:
                        eventType = EventType.NodeCreated;
                        path = event.getData().getPath();
                        content = event.getData().getData() == null ? "" : new String(event.getData().getData(), CHARSET);
                        break;
                    case NODE_UPDATED:
                        eventType = EventType.NodeDataChanged;
                        path = event.getData().getPath();
                        content = event.getData().getData() == null ? "" : new String(event.getData().getData(), CHARSET);
                        break;
                    case NODE_REMOVED:
                        path = event.getData().getPath();
                        eventType = EventType.NodeDeleted;
                        break;
                    case INITIALIZED:
                        eventType = EventType.INITIALIZED;
                        break;
                    case CONNECTION_LOST:
                        eventType = EventType.CONNECTION_LOST;
                        break;
                    case CONNECTION_RECONNECTED:
                        eventType = EventType.CONNECTION_RECONNECTED;
                        break;
                    case CONNECTION_SUSPENDED:
                        eventType = EventType.CONNECTION_SUSPENDED;
                        break;
                    default:
                        break;
                }
                dataListener.dataChanged(path, content, eventType);
            }
        }
    }

//    public static void main(String[] args){
//
//        ZkConfig  zkConfig = new ZkConfig();
//
//        CuratorZookeeperClient  curatorZookeeperClient = new  CuratorZookeeperClient(zkConfig);
//        curatorZookeeperClient.
//
//    }


}
