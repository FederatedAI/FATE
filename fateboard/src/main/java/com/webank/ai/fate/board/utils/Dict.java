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
package com.webank.ai.fate.board.utils;

public class Dict {

    static public final String ID = "id";
    static public final String NAME = "name";
    static public final String DESC = "desc";
    static public final String JOBID = "job_id";
    static public final String RETCODE = "retcode";
    static public final String DATA = "data";
    static public final String JOB = "job";
    static public final String DATASET = "dataset";
    static public final String COMPONENT_NAME = "component_name";
    static public final String ROLE = "role";
    static public final String PARTY_ID = "party_id";

    static public final String DEPENDENCY_DATA = "dependency_data";
    static public final String DATAVIEW_DATA = "dataview_data";


    static public final String METRIC_NAMESPACE = "metric_namespace";
    static public final String METRIC_NAME = "metric_name";
    static public final String STATUS = "status";
    static public final String COMPONENT_LIST = "component_list";


    static public final String CREATE_TIME = "create_time";
    static public final String SSH_CONFIG_FILE = "ssh_config_file";
    static public final String LOG_LINE_NUM = "lineNum";
    static public final String LOG_CONTENT = "content";
    static public final String JOB_PROCESS = "process";
    static public final String JOB_DURATION = "duration";
    static public final String JOB_STATUS = "status";


    static public final String URL_COPONENT_METRIC_DATA = "/v1/tracking/component/metric_data";
    static public final String URL_COPONENT_METRIC = "/v1/tracking/component/metrics";
    static public final String URL_COPONENT_PARAMETERS = "/v1/tracking/component/parameters";
    static public final String URL_DAG_DEPENDENCY = "/v1/pipeline/dag/dependency";
    static public final String URL_OUTPUT_MODEL = "/v1/tracking/component/output/model";
    static public final String URL_OUTPUT_DATA = "/v1/tracking/component/output/data";
    static public final String URL_JOB_DATAVIEW = "/v1/tracking/job/data_view";
    static public final String URL_JOB_STOP = "/v1/job/stop";
    static public final String REMOTE_RETURN_MSG = "retmsg";

    static public final String SSH_IP = "ip";
    static public final String SSH_USER = "user";
    static public final String SSH_PASSWORD = "password";
    static public final String SSH_PORT = "port";

}
