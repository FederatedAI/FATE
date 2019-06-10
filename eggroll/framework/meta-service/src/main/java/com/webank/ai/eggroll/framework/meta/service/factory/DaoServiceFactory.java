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

package com.webank.ai.eggroll.framework.meta.service.factory;


import com.webank.ai.eggroll.framework.meta.service.dao.generated.mapper.*;
import com.webank.ai.eggroll.framework.meta.service.dao.generated.model.*;
import com.webank.ai.eggroll.framework.meta.service.service.impl.GenericDaoService;
import com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Node;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.ApplicationContext;
import org.springframework.stereotype.Component;

@Component
public class DaoServiceFactory {
    @Autowired
    private ApplicationContext applicationContext;

    public GenericDaoService<com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Dtable, DtableExample, Long> createDtableDaoService() {
        GenericDaoService<com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Dtable, DtableExample, Long> result = applicationContext.getBean(GenericDaoService.class);
        result.init(com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Dtable.class, DtableExample.class, Long.class, DtableMapper.class);

        return result;
    }

    public GenericDaoService<com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Fragment, FragmentExample, Long> createFragmentDaoService() {
        GenericDaoService<com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Fragment, FragmentExample, Long> result = applicationContext.getBean(GenericDaoService.class);
        result.init(com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Fragment.class, FragmentExample.class, Long.class, FragmentMapper.class);

        return result;
    }

    public GenericDaoService<com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Node, NodeExample, Long> createNodeDaoService() {
        GenericDaoService<com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Node, NodeExample, Long> result = applicationContext.getBean(GenericDaoService.class);
        result.init(Node.class, NodeExample.class, Long.class, NodeMapper.class);

        return result;
    }

    public GenericDaoService<com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Task, TaskExample, Long> createTaskDaoService() {
        GenericDaoService<com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Task, TaskExample, Long> result = applicationContext.getBean(GenericDaoService.class);
        result.init(com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Task.class, TaskExample.class, Long.class, TaskMapper.class);

        return result;
    }

    public GenericDaoService<com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Result, ResultExample, Long> createResultDaoService() {
        GenericDaoService<com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Result, ResultExample, Long> result = applicationContext.getBean(GenericDaoService.class);
        result.init(com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Result.class, ResultExample.class, Long.class, ResultMapper.class);

        return result;
    }
}
