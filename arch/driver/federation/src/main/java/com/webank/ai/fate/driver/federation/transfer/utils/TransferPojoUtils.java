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

package com.webank.ai.fate.driver.federation.transfer.utils;

import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;
import com.webank.ai.eggroll.api.core.BasicMeta;
import com.webank.ai.fate.api.driver.federation.Federation;
import com.webank.ai.eggroll.core.constant.StringConstants;
import org.apache.commons.lang3.StringUtils;
import org.springframework.context.annotation.Scope;
import org.springframework.stereotype.Component;

import java.util.List;

@Component
@Scope("prototype")
public class TransferPojoUtils {
    private static final String DELIMITER = StringConstants.DASH;

    public String generateTransferId(Federation.TransferMeta transferMeta) {
        Preconditions.checkNotNull(transferMeta, "transferMeta cannot be null");

        BasicMeta.Job job = transferMeta.getJob();
        String tag = transferMeta.getTag();
        Federation.Party src = transferMeta.getSrc();
        Federation.Party dst = transferMeta.getDst();

        Preconditions.checkNotNull(job, "job instance cannot be null");
        Preconditions.checkArgument(StringUtils.isNotBlank(job.getJobId()), "Job id cannot be blank");
        Preconditions.checkArgument(StringUtils.isNotBlank(tag), "tag cannot be blank");

        List<String> elements = Lists.newLinkedList();
        elements.add(job.getJobId());
        elements.add(job.getName());
        elements.add(tag);
        elements.add(src.getPartyId());
        elements.add(src.getName());
        elements.add(dst.getPartyId());
        elements.add(dst.getName());

        return String.join(DELIMITER, elements);
    }

/*    public String generateTransferId(BasicMeta.Job job, String tag) {
        Preconditions.checkNotNull(job, "job instance cannot be null");
        Preconditions.checkArgument(StringUtils.isNotBlank(job.getJobId()), "Job id cannot be blank");
        Preconditions.checkArgument(StringUtils.isNotBlank(tag), "tag cannot be blank");

        StringBuilder builder = new StringBuilder();
        builder.append(job.getJobId())
                .append(StringConstants.DASH)
                .append(tag);

        return builder.toString();
    }*/

    public String generateSendObjectKey(Federation.TransferMeta transferMeta) {
        Preconditions.checkNotNull(transferMeta, "transferMeta cannot be null");
        StringBuilder sb = new StringBuilder();
        sb.append(transferMeta.getTag())
                .append(DELIMITER)
                .append(transferMeta.getDataDesc().getStorageLocator().getName());

        return sb.toString();
    }
}
