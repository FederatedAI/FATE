package com.webank.ai.fate.board.utils;

import com.alibaba.fastjson.JSON;
import com.alibaba.fastjson.JSONObject;
import com.webank.ai.fate.board.global.ResponseResult;

public class ResponseUtil {


    public static ResponseResult buildResponse(String result, String dataName) {


        JSONObject resultObject = JSON.parseObject(result);

        Integer retcode = resultObject.getInteger(Dict.RETCODE);

        JSONObject data = null;
        if (dataName != null) {

            data = resultObject.getJSONObject(Dict.DATA);
        } else {
            data = resultObject;
        }

        String msg = resultObject.getString(Dict.REMOTE_RETURN_MSG);

        return new ResponseResult<>(retcode, msg, data);

    }

}
