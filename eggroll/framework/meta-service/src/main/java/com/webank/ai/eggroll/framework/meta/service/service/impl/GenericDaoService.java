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

import com.webank.ai.eggroll.core.utils.ErrorUtils;
import com.webank.ai.eggroll.framework.meta.service.service.BaseDaoService;
import org.apache.commons.lang3.reflect.MethodUtils;
import org.apache.ibatis.session.RowBounds;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.ApplicationContext;
import org.springframework.context.annotation.Scope;
import org.springframework.stereotype.Service;

import java.lang.reflect.InvocationTargetException;
import java.util.List;

/**
 * @param <M> Model
 * @param <E> Example
 * @param <P> Primary key
 */
@Service(value = "BaseDaoService")
@Scope("prototype")
// todo: check to prevent being injected
public class GenericDaoService<M, E, P> implements BaseDaoService<M, E, P> {
    private static final Integer ZERO = new Integer(0);
    private static final Logger LOGGER = LogManager.getLogger();
    @Autowired
    private ApplicationContext applicationContext;
    @Autowired
    private ErrorUtils errorUtils;
    private Class recordClass;
    private Class exampleClass;
    private Class primaryKeyClass;
    private Object mapper;
    private Class mapperClass;

    public void init(Class recordClass, Class exampleClass, Class primaryKeyClass, Class mapperClass) {
        this.recordClass = recordClass;
        this.exampleClass = exampleClass;
        this.primaryKeyClass = primaryKeyClass;
        this.mapperClass = mapperClass;

        this.mapper = applicationContext.getBean(mapperClass);
    }

    @Override
    public Integer insertSelective(M record) {
        return (Integer) invokeMethodInternal("insertSelective", record);
    }

    @Override
    public Integer updateByPrimaryKey(M record) {
        return (Integer) invokeMethodInternal("updateByPrimaryKey", record);
    }

    @Override
    public Integer updateByPrimaryKeySelective(Object record) {
        return (Integer) invokeMethodInternal("updateByPrimaryKeySelective", record);
    }

    @Override
    public Integer updateByExampleSelective(M record, E example) {
        return (Integer) invokeMethodInternal("updateByExampleSelective", record, example);
    }

    @Override
    public Integer deleteByPrimaryKey(P primaryKey) {
        return (Integer) invokeMethodInternal("deleteByPrimaryKey", primaryKey);
    }

    @Override
    public Integer deleteByExample(E example) {
        return (Integer) invokeMethodInternal("deleteByExample", example);
    }

    @Override
    public List<M> selectByExample(E example) {
        return (List<M>) invokeMethodInternal("selectByExample", example);
    }

    @Override
    public List<M> selectByExampleWithRowbounds(E example, RowBounds rowBounds) {
        return (List<M>) invokeMethodInternal("selectByExampleWithRowbounds", example, rowBounds);
    }

    @Override
    public M selectByPrimaryKey(P primaryKey) {
        return (M) invokeMethodInternal("selectByPrimaryKey", primaryKey);
    }

    private Object invokeMethodInternal(String methodName, Object... args) {
        Object result = null;

        try {
            result = MethodUtils.invokeMethod(mapper, methodName, args);
        } catch (NoSuchMethodException | IllegalAccessException e) {
            throw new IllegalStateException("should not get here", e);
        } catch (InvocationTargetException e) {
            // LOGGER.error(errorUtils.getStackTrace(e));
            throw new RuntimeException(e);
        }

        return result;
    }
}
