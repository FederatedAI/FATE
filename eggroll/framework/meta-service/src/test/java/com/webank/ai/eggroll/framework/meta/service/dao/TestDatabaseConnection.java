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
import org.junit.Test;
import org.junit.runner.RunWith;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.test.context.ContextConfiguration;
import org.springframework.test.context.junit4.SpringJUnit4ClassRunner;

import java.util.List;

@RunWith(SpringJUnit4ClassRunner.class)
@ContextConfiguration(locations = {"classpath*:applicationContext-meta-service.xml"})
public class TestDatabaseConnection {
    @Autowired
    private DtableMapper dtableMapper;

    @Test
    public void testDatabaseSelect() {
        DtableExample dtablesExample = new DtableExample();
        dtablesExample.createCriteria().andTableIdEqualTo(1L);

        List<Dtable> dtables = dtableMapper.selectByExample(dtablesExample);
        System.out.println(dtables.size());

        Dtable dtable = dtables.get(0);

        System.out.println(dtable);
    }

    @Test
    public void testDatabaseCreate() {
        Dtable record = new Dtable();
        record.setNamespace("ns");
        record.setTableName("tn");
        record.setTableType("tt");
        record.setTotalFragments(0);
        record.setSerdes("normal");
        record.setStatus("unknown");
        record.setStorageVersion(1);

        dtableMapper.insertSelective(record);
    }
}
