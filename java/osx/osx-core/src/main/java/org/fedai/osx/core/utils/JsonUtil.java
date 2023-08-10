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

package org.fedai.osx.core.utils;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.google.gson.JsonObject;
import com.google.protobuf.InvalidProtocolBufferException;
import com.google.protobuf.MessageOrBuilder;
import org.apache.commons.lang3.StringUtils;

import java.io.IOException;

public class JsonUtil {

    private static ObjectMapper mapper = new ObjectMapper();

    static {
        mapper.setSerializationInclusion(JsonInclude.Include.NON_NULL);
        mapper.configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);
        mapper.configure(SerializationFeature.FAIL_ON_EMPTY_BEANS, false);
    }

    public static String object2Json(Object o) {
        if (o == null) {
            return null;
        }
        String s = "";
        try {
            s = mapper.writeValueAsString(o);
        } catch (IOException e) {

        }
        return s;
    }

    public static <T> T json2Object(String json, Class<T> c) {
        if (StringUtils.isBlank(json)) {
            return null;
        }
        T t = null;
        try {
            t = mapper.readValue(json, c);
        } catch (IOException igore) {

        }
        return t;
    }

    public static <T> T json2Object(byte[] json, Class<T> c) {
        T t = null;
        try {
            t = mapper.readValue(json, c);

        } catch (IOException e) {
            e.printStackTrace();
        }
        return t;
    }

    public static <T> T json2List(String json, TypeReference<T> typeReference) {
        if (StringUtils.isBlank(json)) {
            return null;
        }
        T result = null;
        try {
            result = mapper.readValue(json, typeReference);
        } catch (IOException igore) {
        }
        return result;
    }

    @SuppressWarnings("unchecked")
    public static <T> T json2Object(String json, TypeReference<T> tr) {
        if (StringUtils.isBlank(json)) {
            return null;
        }
        T t = null;
        try {
            t = (T) mapper.readValue(json, tr);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return (T) t;
    }

//    public static JsonObject object2JsonObject(Object source){
//        String json = object2Json(source);
//        return JsonParser.parseString(json).getAsJsonObject();
//    }

    public static <T> T json2Object(JsonObject source, Class<T> clazz) {
        String json = source.toString();
        return json2Object(json, clazz);
    }

    public static <T> T object2Objcet(Object source, Class<T> clazz) {
        String json = object2Json(source);
        return json2Object(json, clazz);
    }

    public static <T> T object2Objcet(Object source, TypeReference<T> tr) {
        String json = object2Json(source);
        return json2Object(json, tr);
    }

    public static String formatJson(String jsonStr) {
        return formatJson(jsonStr, "\t");
    }

    /***
     * format json string
     */
    public static String formatJson(String jsonStr, String formatChar) {
        if (null == jsonStr || "".equals(jsonStr)) return "";
        jsonStr = jsonStr.replace("\\n", "");
        StringBuilder sb = new StringBuilder();
        char last;
        char current = '\0';
        int indent = 0;
        boolean isInQuotationMarks = false;
        for (int i = 0; i < jsonStr.length(); i++) {
            last = current;
            current = jsonStr.charAt(i);
            switch (current) {
                case '"':
                    if (last != '\\') {
                        isInQuotationMarks = !isInQuotationMarks;
                    }
                    sb.append(current);
                    break;
                case '{':
                case '[':
                    sb.append(current);
                    if (!isInQuotationMarks) {
                        sb.append('\n');
                        indent++;
                        addIndentTab(sb, indent, formatChar);
                    }
                    break;
                case '}':
                case ']':
                    if (!isInQuotationMarks) {
                        sb.append('\n');
                        indent--;
                        addIndentTab(sb, indent, formatChar);
                    }
                    sb.append(current);
                    break;
                case ',':
                    sb.append(current);
                    if (last != '\\' && !isInQuotationMarks) {
                        sb.append('\n');
                        addIndentTab(sb, indent, formatChar);
                    }
                    break;
                case ' ':
                    if (',' != jsonStr.charAt(i - 1)) {
                        sb.append(current);
                    }
                    break;
                case '\\':
                    sb.append("\\");
                    break;
                default:
                    sb.append(current);
            }
        }

        return sb.toString();
    }

    private static void addIndentTab(StringBuilder sb, int indent, String formatChar) {
        for (int i = 0; i < indent; i++) {
            sb.append(formatChar);
        }
    }

    public static String pbToJson(MessageOrBuilder message) {
        try {
            return com.google.protobuf.util.JsonFormat.printer().print(message);
        } catch (InvalidProtocolBufferException e) {
            e.printStackTrace();
        }
        return "";
    }


    public static void main(String[] args) {
        String s = JsonUtil.formatJson("{\"route_table\":{\"default\":{\"default\":[{\"ip\":\"127.0.0.1\",\"port\":9999,\"useSSL\":false}]},\"10000\":{\"default\":[{\"ip\":\"127.0.0.1\",\"port\":8889}],\"serving\":[{\"ip\":\"127.0.0.1\",\"port\":8080}]},\"123\":[{\"host\":\"127.0.0.1\",\"port\":8888,\"useSSL\":false,\"negotiationType\":\"\",\"certChainFile\":\"\",\"privateKeyFile\":\"\",\"caFile\":\"\"}]},\"permission\":{\"default_allow\":true}}");
        System.out.println(s);

    }

}
