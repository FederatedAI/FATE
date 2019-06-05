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

package com.webank.ai.eggroll.framework.meta.service.service;

import org.apache.ibatis.session.RowBounds;

import java.util.List;

/**
 * @param <M> Model
 * @param <E> Example
 * @param <P> Primary key
 */
public interface BaseDaoService<M, E, P> {
    public Integer insertSelective(M record);

    public Integer updateByPrimaryKey(M record);

    public Integer updateByExampleSelective(M record, E example);

    public Integer deleteByPrimaryKey(P primaryKey);

    public Integer deleteByExample(E example);

    public List<M> selectByExample(E example);

    public List<M> selectByExampleWithRowbounds(E example, RowBounds rowBounds);

    public M selectByPrimaryKey(P primaryKey);

    public Integer updateByPrimaryKeySelective(Object record);
}
