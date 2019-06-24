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

package com.webank.ai.eggroll.framework.meta.service.dao;

import com.webank.ai.eggroll.framework.meta.service.dao.generated.mapper.DtableMapper;
import com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Dtable;
import com.webank.ai.eggroll.framework.meta.service.dao.generated.model.DtableExample;
import com.webank.ai.eggroll.framework.meta.service.factory.DaoServiceFactory;
import com.webank.ai.eggroll.framework.meta.service.service.impl.GenericDaoService;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.test.context.ContextConfiguration;
import org.springframework.test.context.junit4.SpringJUnit4ClassRunner;

import java.lang.reflect.ParameterizedType;
import java.lang.reflect.Type;

@RunWith(SpringJUnit4ClassRunner.class)
@ContextConfiguration(locations = {"classpath*:applicationContext-meta-service.xml"})
public class TestDatabaseGenericOperation {
    @Autowired
    private GenericDaoService<Dtable, DtableExample, Long> dtableDaoService;
    @Autowired
    private DaoServiceFactory daoServiceFactory;
    @Autowired
    private DtableMapper dtableMapper;

    @Test
    public void testSelect() {
        DtableExample dtablesExample = new DtableExample();
        dtablesExample.createCriteria().andTableIdEqualTo(1L);

        GenericDaoService<Dtable, DtableExample, Long> testDaoserviceFactoryTest = daoServiceFactory.createDtableDaoService();

        Dtable dtable = testDaoserviceFactoryTest.selectByPrimaryKey(1L);

        System.out.println(dtable);
    }

    @Test
    public void testGenericCreate() {
        Dtable record = new Dtable();

        record.setNamespace("generic_ns");
        record.setTableName("generic_tn");
        record.setTableType("generic_tt");
        record.setTotalFragments(0);
        record.setSerdes("generic_normal");
        record.setStatus("generic_unknown");
        record.setStorageVersion(1);

        GenericDaoService<Dtable, DtableExample, Long> testDaoserviceFactoryTest = daoServiceFactory.createDtableDaoService();

        testDaoserviceFactoryTest.insertSelective(record);
    }

    @Test
    public void printGenericArgumentsOfInstance() {
        ParameterizedType parameterizedType = (ParameterizedType) dtableDaoService.getClass().getGenericInterfaces()[0];
        Type[] actualTypeArguments = parameterizedType.getActualTypeArguments();
        for (Type actualTypeArgument : actualTypeArguments) {
            System.out.println(actualTypeArgument);
        }
    }
}
