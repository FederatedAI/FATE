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

package com.webank.ai.eggroll.framework.meta.service.service.impl;

import com.webank.ai.eggroll.api.core.BasicMeta;
import com.webank.ai.eggroll.core.api.grpc.server.GrpcServerWrapper;
import com.webank.ai.eggroll.core.error.exception.CrudException;
import com.webank.ai.eggroll.core.factory.CallMetaModelFactory;
import com.webank.ai.eggroll.core.helper.ParamValidationHelper;
import com.webank.ai.eggroll.core.serdes.impl.ByteStringSerDesHelper;
import com.webank.ai.eggroll.core.utils.ErrorUtils;
import com.webank.ai.eggroll.core.utils.ToStringUtils;
import com.webank.ai.eggroll.framework.meta.service.factory.DaoServiceFactory;
import com.webank.ai.eggroll.framework.meta.service.service.CrudServerProcessor;
import com.webank.ai.eggroll.framework.meta.service.service.GrpcCrudService;
import io.grpc.stub.StreamObserver;
import org.apache.commons.lang3.reflect.MethodUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Scope;
import org.springframework.stereotype.Service;

import java.lang.reflect.InvocationTargetException;

@Service(value = "grpcCrudService")
@Scope("prototype")
public class DefaultStrategyGrpcCrudService implements GrpcCrudService {
    @Autowired
    private ByteStringSerDesHelper byteStringSerDesHelper;
    @Autowired
    private DaoServiceFactory daoServiceFactory;
    @Autowired
    private CallMetaModelFactory callMetaModelFactory;
    @Autowired
    private ToStringUtils toStringUtils;
    @Autowired
    private ErrorUtils errorUtils;
    @Autowired
    private ParamValidationHelper paramValidationHelper;
    @Autowired
    private GrpcServerWrapper grpcServerWrapper;

    private GenericDaoService genericDaoService;
    private Class recordClass;

    @Override
    public void init(Class recordClass) {
        this.recordClass = recordClass;

        if (genericDaoService == null) {
            String methodName = "create" + recordClass.getSimpleName() + "DaoService";
            try {
                genericDaoService = (GenericDaoService) MethodUtils.invokeMethod(daoServiceFactory, methodName);
            } catch (NoSuchMethodException | IllegalAccessException | InvocationTargetException e) {
                throw new IllegalStateException(e);
            }
        }
    }

    @Override
    public GenericDaoService getGenericDaoService() {
        return genericDaoService;
    }

    @Override
    // todo: separate createIfNotExists and create
    public void create(BasicMeta.CallRequest request, StreamObserver<BasicMeta.CallResponse> responseObserver) {
        processCrudRequest(request, responseObserver, new CrudServerProcessor<Integer>() {
            @Override
            public Integer process(Object record) {
                return genericDaoService.insertSelective(record);
/*                Integer result = null;
                try {
                    result = genericDaoService.insertSelective(record);
                } catch (RuntimeException e) {
                    Dtable dtable = (Dtable) record;

                    DtableExample dtableExample = new DtableExample();
                    dtableExample.createCriteria().andNamespaceEqualTo(dtable.getNamespace())
                            .andTableNameEqualTo(dtable.getTableName())
                            .andStatusEqualTo(DtableStatus.NORMAL.name());

                    List<Dtable> dtableList = genericDaoService.selectByExample(dtableExample);
                    if (dtableList.isEmpty()) {
                        throw e;
                    } else {
                        result = dtableList.size();
                    }
                }

                return result;*/
            }

            @Override
            public boolean isValid(Integer result) {
                return result != null && result > 0;
            }

            @Override
            public Object pickResult(Object originalRecord, Object callResult) {
                return originalRecord;
            }
        });
    }

    @Override
    // todo: combine create and update and select code when have time
    public void update(BasicMeta.CallRequest request, StreamObserver<BasicMeta.CallResponse> responseObserver) {
        processCrudRequest(request, responseObserver, new CrudServerProcessor<Integer>() {
            @Override
            public Integer process(Object record) {
                return genericDaoService.updateByPrimaryKeySelective(record);
            }

            @Override
            public boolean isValid(Integer result) {
                return result != null && result > 0;
            }

            @Override
            public Object pickResult(Object originalRecord, Object callResult) {
                return originalRecord;
            }
        });
    }

    @Override
    public void createOrUpdate(BasicMeta.CallRequest request, StreamObserver<BasicMeta.CallResponse> responseObserver) {
        processCrudRequest(request, responseObserver, new CrudServerProcessor<Integer>() {
            @Override
            public Integer process(Object record) {
                int insertRowsAffected = genericDaoService.insertSelective(record);
                if (insertRowsAffected == 1) {
                    return insertRowsAffected;
                }

                int updateRowsAffected = genericDaoService.updateByPrimaryKeySelective(record);

                return updateRowsAffected;
            }

            @Override
            public boolean isValid(Integer result) {
                return result != null && result > 0;
            }

            @Override
            public Object pickResult(Object originalRecord, Object callResult) {
                return originalRecord;
            }
        });
    }

    @Override
    // todo: combine create and update and select code when have time
    public void getById(BasicMeta.CallRequest request, StreamObserver responseObserver) {
        processCrudRequest(request, responseObserver, new CrudServerProcessor<Object>() {
            @Override
            public Object process(Object record) {
                return genericDaoService.selectByPrimaryKey(record);
            }

            @Override
            public boolean isValid(Object result) {
                return true;
            }

            @Override
            public Object pickResult(Object originalRecord, Object callResult) {
                return callResult;
            }
        });
    }


    public <T> void processCrudRequestInternal(BasicMeta.CallRequest request,
                                               StreamObserver response,
                                               CrudServerProcessor<T> crudServerProcessor) {
        BasicMeta.CallResponse result = null;

        try {
            paramValidationHelper.validate(request);

            BasicMeta.Data requestData = request.getParam();

            Object record = byteStringSerDesHelper.deserialize(requestData.getData(), Class.forName(requestData.getType()));

            T callResult = (T) crudServerProcessor.process(record);

            if (crudServerProcessor.isValid(callResult)) {
                result = callMetaModelFactory.createNormalCallResponse(crudServerProcessor.pickResult(record, callResult));
            } else {
                result = callMetaModelFactory.createErrorCallResponse(
                        102, "Failed to perform" + recordClass.getSimpleName() + " crud operation in database", record);
            }

            response.onNext(result);
            response.onCompleted();
        } catch (Exception e) {
            response.onError(errorUtils.toGrpcRuntimeException(e));
        }
    }

    @Override
    public <T> void processCrudRequest(BasicMeta.CallRequest request,
                                       StreamObserver responseObserver,
                                       CrudServerProcessor<T> crudServerProcessor) {
        grpcServerWrapper.wrapGrpcServerRunnable(responseObserver, () -> {
            BasicMeta.CallResponse result = null;
            paramValidationHelper.validate(request);

            BasicMeta.Data requestData = request.getParam();


            Object record = byteStringSerDesHelper.deserialize(requestData.getData(), Class.forName(requestData.getType()));
            Object callResult = processCrudRequest(record, crudServerProcessor);

            result = callMetaModelFactory.createNormalCallResponse(callResult);

            responseObserver.onNext(result);
            responseObserver.onCompleted();
        });
    }

    @Override
    public <T> Object processCrudRequest(Object record, CrudServerProcessor<T> crudServerProcessor) throws CrudException {

        T callResult = (T) crudServerProcessor.process(record);

        if (crudServerProcessor.isValid(callResult)) {
            return crudServerProcessor.pickResult(record, callResult);
        } else {
            throw new CrudException(102, "Failed to perform " + recordClass.getSimpleName() + " crud operation in database");
        }
    }

}
