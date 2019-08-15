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
package com.webank.ai.fate.board.dao;

import com.webank.ai.fate.board.pojo.Job;
import com.webank.ai.fate.board.pojo.JobExample;
import com.webank.ai.fate.board.pojo.JobKey;
import com.webank.ai.fate.board.pojo.JobWithBLOBs;
import org.apache.ibatis.annotations.Param;

import java.util.List;

public interface JobMapper {

    List<JobWithBLOBs> selectByPage(@Param(value = "startIndex") long startIndex,
                                    @Param(value = "pageSize") long pageSize);

    long countByExample(JobExample example);

    int deleteByExample(JobExample example);

    int deleteByPrimaryKey(JobKey key);

    int insert(JobWithBLOBs record);

    int insertSelective(JobWithBLOBs record);

    List<JobWithBLOBs> selectByExampleWithBLOBs(JobExample example);

    List<Job> selectByExample(JobExample example);

    JobWithBLOBs selectByPrimaryKey(JobKey key);

    int updateByExampleSelective(@Param("record") JobWithBLOBs record, @Param("example") JobExample example);

    int updateByExampleWithBLOBs(@Param("record") JobWithBLOBs record, @Param("example") JobExample example);

    int updateByExample(@Param("record") Job record, @Param("example") JobExample example);

    int updateByPrimaryKeySelective(JobWithBLOBs record);

    int updateByPrimaryKeyWithBLOBs(JobWithBLOBs record);

    int updateByPrimaryKey(Job record);

}