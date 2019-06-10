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

package com.webank.ai.eggroll.framework.meta.service.api.grpc.server;

import com.google.common.collect.Lists;
import com.webank.ai.eggroll.api.core.BasicMeta;
import com.webank.ai.eggroll.api.framework.meta.service.StorageMetaServiceGrpc;
import com.webank.ai.eggroll.core.api.grpc.server.GrpcServerWrapper;
import com.webank.ai.eggroll.core.error.exception.CrudException;
import com.webank.ai.eggroll.core.factory.CallMetaModelFactory;
import com.webank.ai.eggroll.core.helper.ParamValidationHelper;
import com.webank.ai.eggroll.core.model.DtableStatus;
import com.webank.ai.eggroll.core.model.FragmentStatus;
import com.webank.ai.eggroll.core.model.NodeStatus;
import com.webank.ai.eggroll.core.model.NodeType;
import com.webank.ai.eggroll.core.serdes.impl.ByteStringSerDesHelper;
import com.webank.ai.eggroll.core.utils.CrudUtils;
import com.webank.ai.eggroll.core.utils.ToStringUtils;
import com.webank.ai.eggroll.framework.meta.service.dao.generated.model.*;
import com.webank.ai.eggroll.framework.meta.service.factory.DaoServiceFactory;
import com.webank.ai.eggroll.framework.meta.service.factory.GrpcCrudServiceFactory;
import com.webank.ai.eggroll.framework.meta.service.service.CrudServerProcessor;
import com.webank.ai.eggroll.framework.meta.service.service.GrpcCrudService;
import com.webank.ai.eggroll.framework.meta.service.service.impl.GenericDaoService;
import com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Node;
import io.grpc.stub.StreamObserver;
import org.apache.commons.lang3.StringUtils;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Scope;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;
import java.util.List;

@Component
@Scope("prototype")
public class StorageMetaServiceImpl extends StorageMetaServiceGrpc.StorageMetaServiceImplBase {
    private static final Logger LOGGER = LogManager.getLogger(StorageMetaServiceImpl.class);
    @Autowired
    private GrpcCrudServiceFactory grpcCrudServiceFactory;
    @Autowired
    private ByteStringSerDesHelper byteStringSerDesHelper;
    @Autowired
    private DaoServiceFactory daoServiceFactory;
    @Autowired
    private CallMetaModelFactory callMetaModelFactory;
    @Autowired
    private ToStringUtils toStringUtils;
    @Autowired
    private ParamValidationHelper paramValidationHelper;
    @Autowired
    private GrpcServerWrapper grpcServerWrapper;
    private GrpcCrudService dtableGrpcCrudService;
    private GrpcCrudService fragmentGrpcCrudService;
    private GrpcCrudService nodeGrpcCrudService;

    @PostConstruct
    public void init() {
        dtableGrpcCrudService = grpcCrudServiceFactory.createGrpcCrudService(com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Dtable.class);
        fragmentGrpcCrudService = grpcCrudServiceFactory.createGrpcCrudService(com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Fragment.class);
        nodeGrpcCrudService = grpcCrudServiceFactory.createGrpcCrudService(com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Node.class);
    }

    @Override
    public void createTable(BasicMeta.CallRequest request, StreamObserver<BasicMeta.CallResponse> responseObserver) {
        LOGGER.info("create table called: {}", toStringUtils.toOneLineString(request));

        grpcServerWrapper.wrapGrpcServerRunnable(responseObserver, () -> {
            dtableGrpcCrudService.create(request, responseObserver);
        });
    }

    @Override
    public void createTableIfAbsent(BasicMeta.CallRequest request, StreamObserver<BasicMeta.CallResponse> responseObserver) {
        LOGGER.info("create table if absent called: {}", toStringUtils.toOneLineString(request));
        dtableGrpcCrudService.processCrudRequest(request, responseObserver, new CrudServerProcessor<com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Dtable>() {
            @Override
            public com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Dtable process(Object record) throws CrudException {
                GenericDaoService genericDaoService = dtableGrpcCrudService.getGenericDaoService();
                com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Dtable result = null;

                if (record == null) {
                    throw new CrudException(100, "input parameter cannot be null");
                }

                com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Dtable dtable = (com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Dtable) record;

                DtableExample example = new DtableExample();
                DtableExample.Criteria criteria = example.createCriteria().andStatusEqualTo(DtableStatus.NORMAL.name());

                String namespace = dtable.getNamespace();
                if (StringUtils.isNotBlank(namespace)) {
                    criteria.andNamespaceEqualTo(namespace);
                }

                String tableName = dtable.getTableName();
                if (StringUtils.isNotBlank(tableName)) {
                    criteria.andTableNameEqualTo(tableName);
                }

                List callResult = genericDaoService.selectByExampleWithRowbounds(example, CrudUtils.ROWBOUNDS_ZERO_TO_ONE);
                List<com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Dtable> selectResult = (List<com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Dtable>) callResult;

                if (!selectResult.isEmpty()) {
                    result = selectResult.get(0);
                } else {
                    try {
                        int rowsAffected = genericDaoService.insertSelective(dtable);
                        if (rowsAffected == 1) {
                            result = dtable;
                        }
                    } catch (RuntimeException e) {
                        callResult = genericDaoService.selectByExampleWithRowbounds(example, CrudUtils.ROWBOUNDS_ZERO_TO_ONE);
                        selectResult = (List<com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Dtable>) callResult;

                        if (!selectResult.isEmpty()) {
                            result = selectResult.get(0);
                        } else {
                            throw e;
                        }
                    }
                }

                return result;
            }

            @Override
            public boolean isValid(com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Dtable result) {
                return true;
            }

            @Override
            public Object pickResult(Object originalRecord, Object callResult) {
                return callResult;
            }
        });
    }

    @Override
    public void updateTable(BasicMeta.CallRequest request, StreamObserver<BasicMeta.CallResponse> responseObserver) {
        LOGGER.info("Updating table: {}", toStringUtils.toOneLineString(request));

        grpcServerWrapper.wrapGrpcServerRunnable(responseObserver, () -> {
            dtableGrpcCrudService.update(request, responseObserver);
        });
    }

    @Override
    public void updateFragment(BasicMeta.CallRequest request, StreamObserver<BasicMeta.CallResponse> responseObserver) {
        LOGGER.info("updateFragment: {}", toStringUtils.toOneLineString(request));

        grpcServerWrapper.wrapGrpcServerRunnable(responseObserver, () -> {
            fragmentGrpcCrudService.update(request, responseObserver);
        });
    }

    @Override
    public void getTableById(BasicMeta.CallRequest request, StreamObserver<BasicMeta.CallResponse> responseObserver) {
        LOGGER.info("getTableById: {}", toStringUtils.toOneLineString(request));

        grpcServerWrapper.wrapGrpcServerRunnable(responseObserver, () -> {
            dtableGrpcCrudService.getById(request, responseObserver);
        });
    }

    @Override
    public void getTable(BasicMeta.CallRequest request, StreamObserver<BasicMeta.CallResponse> responseObserver) {
        LOGGER.info("getTable: {}", toStringUtils.toOneLineString(request));

        dtableGrpcCrudService.processCrudRequest(request, responseObserver, new CrudServerProcessor<com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Dtable>() {
            @Override
            public com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Dtable process(Object record) throws CrudException {
                GenericDaoService genericDaoService = dtableGrpcCrudService.getGenericDaoService();
                com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Dtable result = null;

                if (record == null) {
                    throw new CrudException(100, "input parameter cannot be null");
                }

                com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Dtable dtable = (com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Dtable) record;

                DtableExample example = new DtableExample();
                DtableExample.Criteria criteria = example.createCriteria().andStatusEqualTo(DtableStatus.NORMAL.name());

                String namespace = dtable.getNamespace();
                if (StringUtils.isNotBlank(namespace)) {
                    criteria.andNamespaceEqualTo(namespace);
                }

                String tableName = dtable.getTableName();
                if (StringUtils.isNotBlank(tableName)) {
                    criteria.andTableNameEqualTo(tableName);
                }

                List callResult = genericDaoService.selectByExampleWithRowbounds(example, CrudUtils.ROWBOUNDS_ZERO_TO_ONE);
                List<com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Dtable> selectResult = (List<com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Dtable>) callResult;

                if (!selectResult.isEmpty()) {
                    result = selectResult.get(0);
                }

                return result;
            }

            @Override
            public boolean isValid(com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Dtable result) {
                return true;
            }

            @Override
            public Object pickResult(Object originalRecord, Object callResult) {
                return callResult;
            }
        });
    }

    @Override
    public void getTables(BasicMeta.CallRequest request, StreamObserver<BasicMeta.CallResponse> responseObserver) {
        LOGGER.info("getTable: {}", toStringUtils.toOneLineString(request));

        dtableGrpcCrudService.processCrudRequest(request, responseObserver, new CrudServerProcessor<List<com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Dtable>>() {
            @Override
            public List<com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Dtable> process(Object record) throws CrudException {
                GenericDaoService genericDaoService = dtableGrpcCrudService.getGenericDaoService();
                com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Dtable result = null;

                if (record == null) {
                    throw new CrudException(100, "input parameter cannot be null");
                }

                com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Dtable dtable = (com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Dtable) record;

                DtableExample example = new DtableExample();
                DtableExample.Criteria criteria = example.createCriteria().andStatusEqualTo(DtableStatus.NORMAL.name());

                String namespace = dtable.getNamespace();
                if (StringUtils.isNotBlank(namespace)) {
                    criteria.andNamespaceEqualTo(namespace);
                }

                String tableName = dtable.getTableName();
                if (StringUtils.isNotBlank(tableName)) {
                    String tableNameMatch = StringUtils.replace(tableName, "*", "%");
                    criteria.andTableNameLikeInsensitive(tableNameMatch);
                }

                String tableType = dtable.getTableType();
                if (StringUtils.isNotBlank(tableType)) {
                    criteria.andTableTypeEqualTo(tableType);
                }

                List callResult = genericDaoService.selectByExample(example);
                List<com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Dtable> selectResult = (List<com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Dtable>) callResult;

                return selectResult;
            }

            @Override
            public boolean isValid(List<com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Dtable> result) {
                return result != null;
            }

            @Override
            public Object pickResult(Object originalRecord, Object callResult) {
                return callResult;
            }
        });
    }

    @Override
    public void getFragmentsByTableId(BasicMeta.CallRequest request, StreamObserver<BasicMeta.CallResponse> responseObserver) {
        LOGGER.info("getFragmentsByTableId: {}", toStringUtils.toOneLineString(request));

        dtableGrpcCrudService.processCrudRequest(request, responseObserver, new CrudServerProcessor<List<com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Fragment>>() {
            @Override
            public List<com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Fragment> process(Object record) throws CrudException {
                GenericDaoService genericDaoService = fragmentGrpcCrudService.getGenericDaoService();
                List<com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Fragment> result = null;

                Long tableId = (Long) record;

                FragmentExample example = new FragmentExample();
                example.createCriteria().andTableIdEqualTo(tableId);

                Object callResult = genericDaoService.selectByExample(example);
                List<com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Fragment> selectResult = (List<com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Fragment>) callResult;

                return selectResult;
            }

            @Override
            public boolean isValid(List<com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Fragment> result) {
                return result != null;
            }

            @Override
            public Object pickResult(Object originalRecord, Object callResult) {
                return callResult;
            }
        });
    }

    @Override
    public void getNodeById(BasicMeta.CallRequest request, StreamObserver<BasicMeta.CallResponse> responseObserver) {
        LOGGER.info("getNodeById called. request: {}", toStringUtils.toOneLineString(request));

        grpcServerWrapper.wrapGrpcServerRunnable(responseObserver, () -> {
            nodeGrpcCrudService.getById(request, responseObserver);
        });
    }

    @Override
    public void getFragmentById(BasicMeta.CallRequest request, StreamObserver<BasicMeta.CallResponse> responseObserver) {
        LOGGER.info("getFragmentById called. request: {}", toStringUtils.toOneLineString(request));

        grpcServerWrapper.wrapGrpcServerRunnable(responseObserver, () -> {
            fragmentGrpcCrudService.getById(request, responseObserver);
        });
    }

    @Override
    public void getStorageNodesByTableId(BasicMeta.CallRequest request, StreamObserver<BasicMeta.CallResponse> responseObserver) {
        LOGGER.info("getStorageNodesByTableId. request: {}", toStringUtils.toOneLineString(request));

        nodeGrpcCrudService.processCrudRequest(request, responseObserver, new CrudServerProcessor<List<com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Node>>() {
            @Override
            public List<com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Node> process(Object record) throws CrudException {
                GenericDaoService nodeDaoService = nodeGrpcCrudService.getGenericDaoService();
                GenericDaoService fragmentDaoService = fragmentGrpcCrudService.getGenericDaoService();

                Long tableId = (Long) record;

                FragmentExample fragmentExample = new FragmentExample();
                fragmentExample.createCriteria().andTableIdEqualTo(tableId);

                List<com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Fragment> fragments = fragmentDaoService.selectByExample(fragmentExample);
                List<Long> nodeIds = Lists.newArrayList();

                for (com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Fragment fragment : fragments) {
                    nodeIds.add(fragment.getNodeId());
                }

                NodeExample nodeExample = new NodeExample();
                nodeExample.createCriteria()
                        .andNodeIdIn(nodeIds)
                        .andStatusEqualTo(NodeStatus.HEALTHY.name());

                List<com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Node> result = nodeDaoService.selectByExample(nodeExample);

                return result;
            }

            @Override
            public boolean isValid(List<com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Node> result) {
                return result != null;
            }

            @Override
            public Object pickResult(Object originalRecord, Object callResult) {
                return callResult;
            }
        });
    }

    @Override
    public void getEggNodeManagerByIp(BasicMeta.CallRequest request, StreamObserver<BasicMeta.CallResponse> responseObserver) {
        LOGGER.info("getComputingNodeManagerByIp. request: {}", toStringUtils.toOneLineString(request));

        nodeGrpcCrudService.processCrudRequest(request, responseObserver, new CrudServerProcessor<com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Node>() {
            @Override
            public com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Node process(Object record) throws CrudException {
                GenericDaoService nodeDaoService = nodeGrpcCrudService.getGenericDaoService();

                com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Node result = null;
                String ip = (String) record;

                NodeExample nodeExample = new NodeExample();
                nodeExample.createCriteria()
                        .andTypeEqualTo(NodeType.EGG.name())
                        .andStatusEqualTo(NodeStatus.HEALTHY.name())
                        .andIpEqualTo(ip);

                List<com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Node> results = nodeDaoService.selectByExampleWithRowbounds(nodeExample, CrudUtils.ROWBOUNDS_ZERO_TO_ONE);

                if (!results.isEmpty()) {
                    result = results.get(0);
                }

                return result;
            }

            @Override
            public boolean isValid(com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Node result) {
                return true;
            }

            @Override
            public Object pickResult(Object originalRecord, Object callResult) {
                return callResult;
            }
        });
    }

    @Override
    public void getNodesByIds(BasicMeta.CallRequest request, StreamObserver<BasicMeta.CallResponse> responseObserver) {
        LOGGER.info("getNodesByIds. request: {}", toStringUtils.toOneLineString(request));

        nodeGrpcCrudService.processCrudRequest(request, responseObserver, new CrudServerProcessor<List<com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Node>>() {
            @Override
            public List<com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Node> process(Object record) throws CrudException {
                GenericDaoService genericDaoService = nodeGrpcCrudService.getGenericDaoService();
                List<com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Node> result = null;

                List<Long> nodeIds = (List<Long>) record;

                NodeExample example = new NodeExample();
                example.createCriteria().andNodeIdIn(nodeIds);

                Object callResult = genericDaoService.selectByExample(example);
                result = (List<com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Node>) callResult;

                return result;
            }

            @Override
            public boolean isValid(List<com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Node> result) {
                return result != null;
            }

            @Override
            public Object pickResult(Object originalRecord, Object callResult) {
                return callResult;
            }
        });
    }

    @Override
    public void getNodesOfStatus(BasicMeta.CallRequest request, StreamObserver<BasicMeta.CallResponse> responseObserver) {
        LOGGER.info("getNodeOfStatus. request: {}", toStringUtils.toOneLineString(request));

        nodeGrpcCrudService.processCrudRequest(request, responseObserver, new GetNodeOfStatusCrudProcessor());
    }

    @Override
    public void createFragmentsForTable(BasicMeta.CallRequest request, StreamObserver<BasicMeta.CallResponse> responseObserver) {
        LOGGER.info("createFragmentsForTable. request: {}", toStringUtils.toOneLineString(request));

        fragmentGrpcCrudService.processCrudRequest(request, responseObserver, new CrudServerProcessor<List<com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Fragment>>() {
            @Override
            public List<com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Fragment> process(Object record) throws CrudException {
                GenericDaoService fragmentDaoService = fragmentGrpcCrudService.getGenericDaoService();
                GenericDaoService nodeDaoService = nodeGrpcCrudService.getGenericDaoService();
                List<com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Fragment> result = Lists.newArrayList();

                NodeExample nodeExample = new NodeExample();
                nodeExample.createCriteria().andStatusEqualTo(NodeStatus.HEALTHY.name())
                        .andTypeEqualTo(NodeType.STORAGE.name());
                List<com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Node> healthyStorageNodes = nodeDaoService.selectByExample(nodeExample);

                if (healthyStorageNodes.size() == 0) {
                    throw new CrudException(400, "No healthy node available");
                }

                com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Dtable dtable = (com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Dtable) record;

                int totalFragments = dtable.getTotalFragments();
                int healthyNodeSize = healthyStorageNodes.size();
                Long tableId = dtable.getTableId();
                if (tableId == null) {
                    throw new CrudException(401, "createFragmentsForTable: tableId cannot be null. table: " + dtable);
                }
                int rowsAffected = 0;
                String defaultStatus = FragmentStatus.BACKUP.name();

                for (int i = 0; i < totalFragments; ++i) {
                    com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Fragment fragment = new com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Fragment();
                    fragment.setFragmentOrder(i);
                    fragment.setTableId(tableId);
                    fragment.setStatus(defaultStatus);

                    com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Node node = healthyStorageNodes.get(i % healthyNodeSize);
                    fragment.setNodeId(node.getNodeId());

                    rowsAffected = fragmentDaoService.insertSelective(fragment);

                    if (rowsAffected == 1) {
                        result.add(fragment);
                    }
                }

                return result;
            }

            @Override
            public boolean isValid(List<com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Fragment> result) {
                return result != null;
            }

            @Override
            public Object pickResult(Object originalRecord, Object callResult) {
                return callResult;
            }
        });
    }


    @Override
    public void getNodes(BasicMeta.CallRequest request, StreamObserver<BasicMeta.CallResponse> responseObserver) {
        LOGGER.info("getNode. request: {}", toStringUtils.toOneLineString(request));

        nodeGrpcCrudService.processCrudRequest(request, responseObserver, new CrudServerProcessor<List<com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Node>>() {
            @Override
            public List<com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Node> process(Object record) throws CrudException {
                GenericDaoService nodeDaoService = nodeGrpcCrudService.getGenericDaoService();

                List<com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Node> result = null;
                if (record == null) {
                    throw new CrudException(100, "input parameter cannot be null");
                }

                com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Node node = (com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Node) record;

                NodeExample nodeExample = new NodeExample();
                NodeExample.Criteria criteria = nodeExample.createCriteria();

                String host = node.getHost();
                if (StringUtils.isNotBlank(host)) {
                    criteria.andHostEqualTo(host);
                }

                String ip = node.getIp();
                if (StringUtils.isNotBlank(ip)) {
                    criteria.andIpEqualTo(ip);
                }

                Integer port = node.getPort();
                if (port != null) {
                    criteria.andPortEqualTo(port);
                }

                String type = node.getType();
                if (StringUtils.isNotBlank(type)) {
                    criteria.andTypeEqualTo(type);
                }

                String status = node.getStatus();
                if (StringUtils.isNotBlank(status)) {
                    criteria.andStatusEqualTo(status);
                }

                List callResult = nodeDaoService.selectByExample(nodeExample);
                result = (List<com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Node>) callResult;

                return result;
            }

            @Override
            public boolean isValid(List<com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Node> result) {
                return result != null;
            }

            @Override
            public Object pickResult(Object originalRecord, Object callResult) {
                return callResult;
            }
        });
    }

    public class GetNodeOfStatusCrudProcessor implements CrudServerProcessor<List<com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Node>> {
        @Override
        public List<com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Node> process(Object record) throws CrudException {
            GenericDaoService genericDaoService = nodeGrpcCrudService.getGenericDaoService();

            List<com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Node> result = null;

            NodeStatus nodeStatus = (NodeStatus) record;

            NodeExample example = new NodeExample();
            example.createCriteria().andStatusEqualTo(nodeStatus.name());

            Object callResult = genericDaoService.selectByExample(example);
            result = (List<com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Node>) callResult;

            return result;
        }

        @Override
        public boolean isValid(List<Node> result) {
            return result != null;
        }

        @Override
        public Object pickResult(Object originalRecord, Object callResult) {
            return callResult;
        }
    }

}
