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

import com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Dtable;
import com.webank.ai.eggroll.framework.meta.service.dao.generated.model.DtableExample;
import com.webank.ai.eggroll.framework.meta.service.factory.DaoServiceFactory;
import com.webank.ai.eggroll.framework.meta.service.service.impl.GenericDaoService;
import org.apache.ibatis.session.RowBounds;
import org.junit.Assert;
import org.junit.FixMethodOrder;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;
import org.junit.runner.RunWith;
import org.junit.runners.MethodSorters;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.test.context.ContextConfiguration;
import org.springframework.test.context.junit4.SpringJUnit4ClassRunner;

import java.util.List;

@RunWith(SpringJUnit4ClassRunner.class)
@ContextConfiguration(locations = {"classpath*:applicationContext-meta-service.xml"})
@FixMethodOrder(MethodSorters.NAME_ASCENDING)
public class TestDaoService {
    @Rule
    public ExpectedException thrown = ExpectedException.none();
    @Autowired
    private DaoServiceFactory daoServiceFactory;
    private GenericDaoService<Dtable, DtableExample, Long> dtableDaoService;
    private Dtable record;
    private DtableExample example;
    private Long primaryKey;
    private volatile boolean inited = false;

    public void init() {
        if (inited) {
            return;
        }
        System.out.println("====== init ======");
        dtableDaoService = daoServiceFactory.createDtableDaoService();

        record = new Dtable();
        example = new DtableExample();

        record.setNamespace("test_ns");
        record.setTableName("test_tn");
        record.setTableType("test_tt");
        record.setTotalFragments(0);
        record.setSerdes("test_normal");
        record.setStatus("test_unknown");
        record.setStorageVersion(0);
        inited = true;
    }

    @Test
    // InsertSelective
    public void test01() {
        init();
        System.out.println("====== test01: InsertSelective ======");
        int rowsAffected = dtableDaoService.insertSelective(record);

        Assert.assertEquals(1, rowsAffected);
    }

    @Test
    // SelectByExample
    public void test02() {
        System.out.println("====== test02: SelectByExample ======");
        example.clear();
        example.createCriteria().andNamespaceEqualTo("test_ns").andTableNameEqualTo("test_tn");

        List<Dtable> results = dtableDaoService.selectByExample(example);

        if (results == null) {
            return;
        }
        Assert.assertTrue(results != null && results.size() > 0);

        Dtable result = results.get(0);
        primaryKey = result.getTableId();
        record.setTableId(primaryKey);
    }

    @Test
    // UpdateByPrimaryKey
    public void test03() {
        System.out.println("====== test03: UpdateByPrimaryKey ======");
        String newStatus = "test_updated";
        record.setStatus(newStatus);

        Integer rowsAffected = dtableDaoService.updateByPrimaryKey(record);

        Assert.assertEquals(1L, rowsAffected.longValue());
    }

    @Test
    // SelectByPrimaryKey
    public void test04() {
        System.out.println("====== test04: SelectByPrimaryKey ======");
        Dtable result = dtableDaoService.selectByPrimaryKey(primaryKey);

        Assert.assertNotNull(result);
    }

    @Test
    // UpdateByExampleSelective
    public void test05() {
        System.out.println("====== test05: UpdateByExampleSelective ======");
        int newStorageVersion = 1;

        example.clear();
        example.createCriteria().andTableIdEqualTo(primaryKey);

        record.setStorageVersion(newStorageVersion);
        Integer rowsAffected = dtableDaoService.updateByExampleSelective(record, example);

        Assert.assertEquals(1L, rowsAffected.longValue());

        Dtable result = dtableDaoService.selectByPrimaryKey(primaryKey);
        Assert.assertEquals(newStorageVersion, result.getStorageVersion().longValue());
    }

    @Test
    // DeleteByPrimaryKey
    public void test06() {
        System.out.println("====== test06: DeleteByPrimaryKey ======");
        int rowsAffected = dtableDaoService.deleteByPrimaryKey(primaryKey);
        Assert.assertEquals(rowsAffected, 1);

        Dtable result = dtableDaoService.selectByPrimaryKey(primaryKey);
        Assert.assertNull(result);
    }

    @Test
    // InsertSelectiveAgain
    public void test07() {
        System.out.println("====== test07: InsertSelectiveAgain ======");
        int rowsAffected = dtableDaoService.insertSelective(record);

        Assert.assertEquals(1, rowsAffected);
    }

    @Test
    // SelectByExampleWithRowBounds
    public void test08() {
        System.out.println("====== test08: SelectByExampleWithRowBounds ======");
        example.clear();
        example.createCriteria().andNamespaceEqualTo("test_ns").andTableNameEqualTo("test_tn");

        RowBounds rowBounds = new RowBounds(0, 10);

        List<Dtable> results = dtableDaoService.selectByExampleWithRowbounds(example, rowBounds);

        if (results == null) {
            return;
        }

        Dtable result = results.get(0);
        primaryKey = result.getTableId();
        record.setTableId(primaryKey);
    }

    @Test
    // DeleteByExample
    public void test09() {
        System.out.println("====== test09: DeleteByExample ======");
        example.clear();
        example.createCriteria().andTableIdEqualTo(primaryKey);

        int rowsAffected = dtableDaoService.deleteByExample(example);
        Assert.assertEquals(1, rowsAffected);
    }

    @Test
    // SelectByPrimaryKeyAgain
    public void test10() {
        System.out.println("====== test10: SelectByPrimaryKeyAgain ======");
        Dtable result = dtableDaoService.selectByPrimaryKey(primaryKey);

        Assert.assertNull(result);
    }
}
