package com.webank.ai.fate.board.services;

import com.webank.ai.fate.board.dao.TaskMapper;
import com.webank.ai.fate.board.pojo.Task;
import com.webank.ai.fate.board.pojo.TaskExample;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class TaskManagerService {
    @Autowired
    TaskMapper taskMapper;


    public String findTaskStatus(String jobId, String componentName) {
        TaskExample taskExample = new TaskExample();
        TaskExample.Criteria criteria = taskExample.createCriteria();
        criteria.andFJobIdEqualTo(jobId);
        criteria.andFComponentNameEqualTo(componentName);
        List<Task> tasks = taskMapper.selectByExample(taskExample);

        if (tasks.size() != 0) {
            Task task = tasks.get(0);
            return task.getfStatus();
        }
        return null;
    }


}
