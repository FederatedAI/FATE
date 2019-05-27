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

package com.webank.ai.fate.eggroll.meta.service.factory;


import com.webank.ai.fate.eggroll.meta.service.dao.generated.mapper.*;
import com.webank.ai.fate.eggroll.meta.service.dao.generated.model.*;
import com.webank.ai.fate.eggroll.meta.service.service.impl.GenericDaoService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.ApplicationContext;
import org.springframework.stereotype.Component;

@Component
public class DaoServiceFactory {
    @Autowired
    private ApplicationContext applicationContext;

    public GenericDaoService<Dtable, DtableExample, Long> createDtableDaoService() {
        GenericDaoService<Dtable, DtableExample, Long> result = applicationContext.getBean(GenericDaoService.class);
        result.init(Dtable.class, DtableExample.class, Long.class, DtableMapper.class);

        return result;
    }

    public GenericDaoService<Fragment, FragmentExample, Long> createFragmentDaoService() {
        GenericDaoService<Fragment, FragmentExample, Long> result = applicationContext.getBean(GenericDaoService.class);
        result.init(Fragment.class, FragmentExample.class, Long.class, FragmentMapper.class);

        return result;
    }

    public GenericDaoService<Node, NodeExample, Long> createNodeDaoService() {
        GenericDaoService<Node, NodeExample, Long> result = applicationContext.getBean(GenericDaoService.class);
        result.init(Node.class, NodeExample.class, Long.class, NodeMapper.class);

        return result;
    }

    public GenericDaoService<Task, TaskExample, Long> createTaskDaoService() {
        GenericDaoService<Task, TaskExample, Long> result = applicationContext.getBean(GenericDaoService.class);
        result.init(Task.class, TaskExample.class, Long.class, TaskMapper.class);

        return result;
    }

    public GenericDaoService<Result, ResultExample, Long> createResultDaoService() {
        GenericDaoService<Result, ResultExample, Long> result = applicationContext.getBean(GenericDaoService.class);
        result.init(Result.class, ResultExample.class, Long.class, ResultMapper.class);

        return result;
    }
}
