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

package com.osx.core.config;

import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
import com.osx.core.constant.DeployMode;
import com.osx.core.constant.Dict;
import com.osx.core.constant.StreamLimitMode;

import java.lang.reflect.Field;
import java.util.Map;
import java.util.Set;

public class MetaInfo {
    public static final long CURRENT_VERSION = 100;
    public static  String PROPERTY_FATE_TECH_PROVIDER = "FATE";
    public static  String PROPERTY_DEFAULT_CLIENT_VERSION="2.X.X";
    public static volatile MasterInfo masterInfo;
    public static int PROPERTY_GRPC_SERVER_MAX_CONCURRENT_CALL_PER_CONNECTION = 1000;
    public static int PROPERTY_GRPC_SERVER_MAX_INBOUND_METADATA_SIZE = 128 << 20;
    public static int PROPERTY_GRPC_SERVER_MAX_INBOUND_MESSAGE_SIZE = (2 << 30) - 1;
    public static int PROPERTY_GRPC_SERVER_FLOW_CONTROL_WINDOW = 128 << 20;
    public static int PROPERTY_GRPC_SERVER_KEEPALIVE_TIME_SEC = 7200;
    public static int PROPERTY_GRPC_SERVER_KEEPALIVE_TIMEOUT_SEC = 3600;
    public static int PROPERTY_GRPC_SERVER_PERMIT_KEEPALIVE_TIME_SEC = 10;
    public static boolean PROPERTY_GRPC_SERVER_KEEPALIVE_WITHOUT_CALLS_ENABLED = true;
    public static int PROPERTY_GRPC_SERVER_MAX_CONNECTION_IDLE_SEC = 86400;
    public static int PROPERTY_GRPC_SERVER_MAX_CONNECTION_AGE_SEC = 86400;
    public static int PROPERTY_GRPC_SERVER_MAX_CONNECTION_AGE_GRACE_SEC = 86400;
    public static int PROPERTY_GRPC_ONCOMPLETED_WAIT_TIMEOUT = 600;




    public static int PROPERTY_GRPC_CLIENT_MAX_CONCURRENT_CALL_PER_CONNECTION = 1000;
    public static int PROPERTY_GRPC_CLIENT_MAX_INBOUND_METADATA_SIZE = 128 << 20;
    public static int PROPERTY_GRPC_CLIENT_MAX_INBOUND_MESSAGE_SIZE = (2 << 30) - 1;
    public static int PROPERTY_GRPC_CLIENT_FLOW_CONTROL_WINDOW = 128 << 20;
    public static int PROPERTY_GRPC_CLIENT_KEEPALIVE_TIME_SEC = 7200;
    public static int PROPERTY_GRPC_CLIENT_KEEPALIVE_TIMEOUT_SEC = 3600;
    public static int PROPERTY_GRPC_CLIENT_PERMIT_KEEPALIVE_TIME_SEC = 10;
    public static boolean PROPERTY_GRPC_CLIENT_KEEPALIVE_WITHOUT_CALLS_ENABLED = true;
    public static int PROPERTY_GRPC_CLIENT_MAX_CONNECTION_IDLE_SEC = 86400;
    public static int PROPERTY_GRPC_CLIENT_MAX_CONNECTION_AGE_SEC = 86400;
    public static int PROPERTY_GRPC_CLIENT_MAX_CONNECTION_AGE_GRACE_SEC = 86400;
    public static int PROPERTY_GRPC_CLIENT_PER_RPC_BUFFER_LIMIT=86400;

    public static int PROPERTY_GRPC_CLIENT_RETRY_BUFFER_SIZE = 86400;



    public static boolean PROPERTY_USE_DIRECT_CACHE = false;
    public static int PROPERTY_TRANSFER_FILE_CACHE_SIZE = 1 << 27;
    public static int PROPERTY_TRANSFER_RETRY_COUNT = 1;
    public static int MAP_FILE_SIZE = 1 << 25;
    public static int PROPERTY_INDEX_MAP_FILE_SIZE = 1 << 21;
    public static Boolean TRANSFER_FATECLOUD_AHTHENTICATION_ENABLED;
    public static Boolean TRANSFER_FATECLOUD_AUTHENTICATION_USE_CONFIG;
    public static String TRANSFER_FATECLOUD_AUTHENTICATION_URI;
    public static String TRANSFER_FATECLOUD_AUTHENTICATION_APPKEY;
    public static String TRANSFER_FATECLOUD_AUTHENTICATION_APPSERCRET;
    public static String TRANSFER_FATECLOUD_AUTHENTICATION_ROLE;
    public static String TRANSFER_FATECLOUD_SECRET_INFO_URL;
    public static String TRANSFER_FATECLOUD_AUTHENTICATION_URL;
    public static String PROPERTY_SERVER_CERTCHAIN_FILE;
    public static String PROPERTY_SERVER_PRIVATEKEY_FILE;
    public static String PROPERTY_SERVER_CA_FILE;
    public static int ROLLSITE_PARTY_ID;
//    public static Integer PROPERTY_PORT;
    public static Integer PROPERTY_GRPC_PORT;
    public static Integer PROPERTY_HTTP_PORT;
    public static Boolean PROPERTY_OPEN_HTTP_SERVER = false;
    public static Boolean PROPERTY_OPEN_GRPC_TLS_SERVER = false;
    public static int     PROPERTY_HTTP_REQUEST_BODY_MAX_SIZE=4096;
    public static String  PROPERTY_HTTP_CONTEXT_PATH="/osx";
    public static String  PROPERTY_HTTP_SERVLET_PATH="/*";
    public static Integer PROPERTY_GRPC_TLS_PORT;
    public static String PROPERTY_ZK_URL;
    public static Boolean PROPERTY_USE_DISRUPTOR = true;
    public static int PROPERTY_STREAM_LIMIT_MAX_TRY_TIME = 3;

    public static String PROPERTY_USER_HOME = "";

    public static Integer PROPERTY_SAMPLE_COUNT = 10;
    public static Integer PROPERTY_INTERVAL_MS = 1000;
    //public static Boolean PROPERTY_USE_QUEUE_MODEL = false;
    public static String PROPERTY_STREAM_LIMIT_MODE = StreamLimitMode.NOLIMIT.name();

    public static Integer PROPERTY_CONSUMER_TIMEOUT = 30000;
    public static Integer PROPERTY_QUEUE_MAX_FREE_TIME;
    public static Integer PROPERTY_MAPPED_FILE_EXPIRE_TIME = 3600 * 1000 * 36;
    public static Integer PROPERTY_MAX_CONSUME_EMPTY_TRY_COUNT = 30;

    public static Integer PROPERTY_MAX_TRANSFER_CACHE_SIZE = 1 << 30;
    public static String PROPERTY_TRANSFER_FILE_PATH_PRE;
    public static String PROPERTY_DEPLOY_MODE = "standalone";
    public static String PROPERTY_TRANSFER_APPLY_CACHE = "/tmp/cachetest";

    public static Set<String> PROPERTY_SELF_PARTY = Sets.newHashSet();//

    public static Integer PROPERTY_APPLY_EXPIRE_TIME = 3000;
    public static Integer PROPERTY_COORDINATOR;
    public static Integer PROPERTY_SERVER_PORT;
    public static String PROPERTY_INFERENCE_SERVICE_NAME;
    public static String PROPERTY_ROUTE_TYPE;
    public static String PROPERTY_ROUTE_TABLE;

    public static String PROPERTY_FLOW_RULE_TABLE;
    public static String PROPERTY_AUTH_FILE;
    public static Boolean PROPERTY_ACL_ENABLE = false;
    public static String PROPERTY_ACL_USERNAME;
    public static String PROPERTY_ACL_PASSWORD;
    public static String PROPERTY_ROOT_PATH;
    public static Boolean PROPERTY_PRINT_INPUT_DATA;
    public static Boolean PROPERTY_PRINT_OUTPUT_DATA;

    public static Boolean PROPERTY_AUTH_OPEN;
    public static String PROPERTY_NEGOTIATIONTYPE;
    public static String PROPERTY_PROXY_GRPC_INTER_CA_FILE;
    public static String PROPERTY_PROXY_GRPC_INTER_CLIENT_CERTCHAIN_FILE;
    public static String PROPERTY_PROXY_GRPC_INTER_CLIENT_PRIVATEKEY_FILE;
    public static String PROPERTY_PROXY_GRPC_INTER_SERVER_CERTCHAIN_FILE;
    public static String PROPERTY_PROXY_GRPC_INTER_SERVER_PRIVATEKEY_FILE;
    public static Integer PROPERTY_ADMIN_HEALTH_CHECK_TIME;
    public static Integer PRPPERTY_QUEUE_MAX_FREE_TIME;
    public static String ROLLSITE_ROUTE_TABLE_KEY;
    public static String ROLLSITE_ROUTE_TABLE_WHITE_LIST;
    public static String ROLLSITE_ROUTE_TABLE_PARTY_ID;
    public static String INSTANCE_ID;

    public static String PROPERTY_EGGROLL_CLUSTER_MANANGER_IP;
    public static Integer PROPERTY_EGGROLL_CLUSTER_MANANGER_PORT;



    public static Integer PROPERTY_CONSUME_SPIN_TIME = 500;

    public static String PROPERTY_CLUSTER_MANAGER_ADDRESS;
    public static Integer PROPERTY_NETTY_CLIENT_TIMEOUT = 3000;

    public static Integer PROPERTY_HEARTBEAT_INTERVAL = 10000;

    public static String PROPERTY_CLUSTER_MANAGER_HOST;
    public static Integer PROPERTY_CLUSTER_MANAGER_PORT;

    public static Boolean PROPERTY_USE_ZOOKEEPER = true;
    public static Boolean PROPERTY_USE_MSG_QUEUE_REPLACE_STREAM = true;

    /**
     * 从连接池中申请连接的超时时间
     */
    public static Integer HTTP_CLIENT_CONFIG_CONN_REQ_TIME_OUT;
    /**
     * 建立连接的超时时间
     */
    public static Integer HTTP_CLIENT_CONFIG_CONN_TIME_OUT;
    /**
     * 等待数据
     */
    public static Integer HTTP_CLIENT_CONFIG_SOCK_TIME_OUT;
    public static Integer HTTP_CLIENT_INIT_POOL_MAX_TOTAL;
    public static Integer HTTP_CLIENT_INIT_POOL_DEF_MAX_PER_ROUTE;
    public static Integer HTTP_CLIENT_INIT_POOL_SOCK_TIME_OUT;
    public static Integer HTTP_CLIENT_INIT_POOL_CONN_TIME_OUT;
    public static Integer HTTP_CLIENT_INIT_POOL_CONN_REQ_TIME_OUT;
    public static Integer HTTP_CLIENT_TRAN_CONN_REQ_TIME_OUT;
    public static Integer HTTP_CLIENT_TRAN_CONN_TIME_OUT;
    public static Integer HTTP_CLIENT_TRAN_SOCK_TIME_OUT;






    public static String getClusterManagerHost() {
        if (PROPERTY_CLUSTER_MANAGER_HOST != null) {
            return PROPERTY_CLUSTER_MANAGER_HOST;
        } else {
            PROPERTY_CLUSTER_MANAGER_HOST = PROPERTY_CLUSTER_MANAGER_ADDRESS.split(":")[0];
            PROPERTY_CLUSTER_MANAGER_PORT = Integer.parseInt(PROPERTY_CLUSTER_MANAGER_ADDRESS.split(":")[1]);
            return PROPERTY_CLUSTER_MANAGER_HOST;
        }
    }

    public static Integer getClusterManagerPort() {
        if (PROPERTY_CLUSTER_MANAGER_PORT != null) {
            return PROPERTY_CLUSTER_MANAGER_PORT;
        } else {
            PROPERTY_CLUSTER_MANAGER_HOST = PROPERTY_CLUSTER_MANAGER_ADDRESS.split(":")[0];
            PROPERTY_CLUSTER_MANAGER_PORT = Integer.parseInt(PROPERTY_CLUSTER_MANAGER_ADDRESS.split(":")[1]);
            return PROPERTY_CLUSTER_MANAGER_PORT;
        }
    }


    public static boolean isCluster() {
        return PROPERTY_DEPLOY_MODE.equals(DeployMode.cluster.name());
    }

    public static Map toMap() {
        Map result = Maps.newHashMap();
        Field[] fields = MetaInfo.class.getFields();

        for (Field field : fields) {
            try {
                if (field.get(MetaInfo.class) != null) {
                    String key = Dict.class.getField(field.getName()) != null ? String.valueOf(Dict.class.getField(field.getName()).get(Dict.class)) : field.getName();
                    result.put(key, field.get(MetaInfo.class));
                }
            } catch (IllegalAccessException | NoSuchFieldException e) {

            }
        }
        return result;
    }

}
