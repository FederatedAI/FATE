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