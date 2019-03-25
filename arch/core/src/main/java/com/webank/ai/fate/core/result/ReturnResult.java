package com.webank.ai.fate.core.result;

import java.lang.reflect.Field;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.HashMap;
import java.util.Map;

public class ReturnResult{
    private int statusCode;
    private String message;
    private String error;
    private Map<String, Object> data;

    public ReturnResult(){
        this.data = new HashMap<>();
    }

    public void setStatusCode(int statusCode) {
        this.statusCode = statusCode;
    }

    public int getStatusCode() {
        return statusCode;
    }

    public void setMessage(String message) {
        this.message = message;
    }

    public String getMessage() {
        return message;
    }

    public void setError(String error) {
        this.error = error;
    }

    public String getError() {
        return error;
    }

    public void setData(String name, String value) {
        this.data.put(name, value);
    }

    public Map<String, Object> getData() {
        return data;
    }

    public static Map<String, Object> toMap(ReturnResult returnResult){
        Map<String, Object> tmp = new HashMap<>();
        try{
            for (Field field : returnResult.getClass().getFields()) {
                String getter = "get" + field.getName().substring(0, 1).toUpperCase() + field.getName().substring(1);
                Method method = returnResult.getClass().getMethod(getter);
                tmp.put(field.getName(), method.invoke(returnResult));
            }
        }
        catch (NoSuchMethodException ex){
        }
        catch (IllegalAccessException ex){

        }
        catch (InvocationTargetException ex){

        }
        return tmp;
    }
}
