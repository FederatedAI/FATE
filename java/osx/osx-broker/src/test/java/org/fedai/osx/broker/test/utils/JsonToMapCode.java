package org.fedai.osx.broker.test.utils;

import com.fasterxml.jackson.core.type.TypeReference;
import org.fedai.osx.core.utils.JsonUtil;
import org.junit.Test;

import java.util.Map;

/**
 * @date 2023/5/29
 * @remark
 */
public class JsonToMapCode {

    String json = "{\"uri\": \"/v2/partner/job/resource/apply\", \"json_body\": {\"role\": \"host\", \"party_id\": \"10008\", \"job_id\": \"202305251708508595320\"}, \"headers\": {}, \"method\": \"POST\"}";

    @Test
    public void run(){
        Map<String ,Object> head = JsonUtil.json2Object(json, new TypeReference<Map<String, Object>>() {
        });
        StringBuffer sb = new StringBuffer();
        head.forEach((k,v)->{
            sb.append("head.put(\"").append(k).append("\",\"").append(v).append("\");").append("\n");
        });
        System.out.println("sb = " + sb);
    }
}
