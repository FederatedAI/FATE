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

import com.webank.ai.eggroll.api.core.BasicMeta;
import com.webank.ai.eggroll.api.framework.meta.service.ClusterMetaServiceGrpc;
import com.webank.ai.eggroll.core.api.grpc.server.GrpcServerWrapper;
import com.webank.ai.eggroll.core.error.exception.CrudException;
import com.webank.ai.eggroll.core.utils.CrudUtils;
import com.webank.ai.eggroll.core.utils.ToStringUtils;
import com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Fragment;
import com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Node;
import com.webank.ai.eggroll.framework.meta.service.dao.generated.model.NodeExample;
import com.webank.ai.eggroll.framework.meta.service.factory.GrpcCrudServiceFactory;
import com.webank.ai.eggroll.framework.meta.service.service.CrudServerProcessor;
import com.webank.ai.eggroll.framework.meta.service.service.GrpcCrudService;
import com.webank.ai.eggroll.framework.meta.service.service.impl.GenericDaoService;
import io.grpc.stub.StreamObserver;
import org.apache.commons.lang3.StringUtils;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.joda.time.DateTime;
import org.joda.time.DateTimeZone;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Scope;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;
import java.util.List;

@Component
@Scope("prototype")
public class ClusterMetaServiceImpl extends ClusterMetaServiceGrpc.ClusterMetaServiceImplBase {
    private static final Logger LOGGER = LogManager.getLogger();
    @Autowired
    private GrpcCrudServiceFactory grpcCrudServiceFactory;
    @Autowired
    private ToStringUtils toStringUtils;
    @Autowired
    private GrpcServerWrapper grpcServerWrapper;
    private GrpcCrudService nodeGrpcCrudService;
    private GrpcCrudService fragmentGrpcCrudService;

    @PostConstruct
    public void init() {
        nodeGrpcCrudService = grpcCrudServiceFactory.createGrpcCrudService(Node.class);
        fragmentGrpcCrudService = grpcCrudServiceFactory.createGrpcCrudService(Fragment.class);
    }

    @Override
    public void registerNode(BasicMeta.CallRequest request, StreamObserver<BasicMeta.CallResponse> responseObserver) {
        LOGGER.info("registerNode. request: {}", toStringUtils.toOneLineString(request));

        nodeGrpcCrudService.processCrudRequest(request, responseObserver, new CrudServerProcessor<Node>() {
            @Override
            public Node process(Object record) throws CrudException {
                GenericDaoService genericDaoService = nodeGrpcCrudService.getGenericDaoService();
                Node result = null;

                if (record == null) {
                    throw new CrudException(100, "input parameter cannot be null");
                }
                Node node = (Node) record;

                NodeExample example = new NodeExample();
                NodeExample.Criteria criteria = example.createCriteria();

                String ip = node.getIp();
                if (ip != null) {
                    criteria.andIpEqualTo(node.getIp());
                } else {
                    String host = node.getHost();
                    if (host != null) {
                        criteria.andHostEqualTo(host);
                    }
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

                Object callResult = genericDaoService.selectByExampleWithRowbounds(example, CrudUtils.ROWBOUNDS_ZERO_TO_ONE);
                List<Node> selectResult = (List<Node>) callResult;
                // Date now = new Date();

                int rowsAffected = -1;
                if (!selectResult.isEmpty()) {
                    result = (Node) selectResult.get(0);
                    result.setUpdatedAt(DateTime.now(DateTimeZone.UTC).toDate());

                    rowsAffected = genericDaoService.updateByPrimaryKey(result);
                } else {
                    rowsAffected = genericDaoService.insertSelective(record);
                    result = (Node) record;
                }

                if (rowsAffected == 1) {
                    return result;
                } else {
                    throw new CrudException(103, "Failed to create or update node");
                }
            }

            @Override
            public boolean isValid(Node result) {
                return true;
            }

            @Override
            public Object pickResult(Object originalRecord, Object callResult) {
                return callResult;
            }
        });
    }

    @Override
    public void deregisterNode(BasicMeta.CallRequest request, StreamObserver<BasicMeta.CallResponse> response) {

    }

    @Override
    public void heartbeat(BasicMeta.CallRequest request, StreamObserver<BasicMeta.CallResponse> response) {

    }
}
