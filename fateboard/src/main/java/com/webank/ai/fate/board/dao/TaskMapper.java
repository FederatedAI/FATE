package com.webank.ai.fate.board.dao;

import com.webank.ai.fate.board.pojo.Task;
import com.webank.ai.fate.board.pojo.TaskExample;
import org.apache.ibatis.annotations.Param;

import java.util.List;

public interface TaskMapper {
    long countByExample(TaskExample example);

    int deleteByExample(TaskExample example);

    int insert(Task record);

    int insertSelective(Task record);

    List<Task> selectByExample(TaskExample example);

    int updateByExampleSelective(@Param("record") Task record, @Param("example") TaskExample example);

    int updateByExample(@Param("record") Task record, @Param("example") TaskExample example);
}