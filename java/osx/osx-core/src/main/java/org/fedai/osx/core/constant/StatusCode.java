/*
 * Copyright 2019 The FATE Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.fedai.osx.core.constant;

public class StatusCode {
    public static final String SUCCESS = "0";
    public static final String QUEUE_INVALID_STATUS = "100";
    public static final String NET_ERROR = "E0000000607";
    public static final String SYSTEM_ERROR = "E0000000500";
    public static final String PARAM_ERROR = "E0000000400";
    public static final String CONFIG_ERROR = "121";
    public static final String TRANSFER_APPLYINFO_SYNC_ERROR = "129";
    public static final String PROXY_ROUTER_ERROR = "130";
    public static final String PROXY_LOAD_ROUTER_TABLE_ERROR = "132";

    public static final String PROXY_UPDATE_ROUTER_TABLE_ERROR = "133";
    public static final String INVALID_RESPONSE = "135";
    public static final String CONSUME_NO_MESSAGE = "136";
    public static final String MESSAGE_PARSE_ERROR = "137";
    public static final String TRANSFER_QUEUE_NOT_FIND = "138";
    public static final String PUT_MESSAGE_ERROR = "139";
    public static final String ACK_INDEX_ERROR = "140";
    public static final String CONSUMER_NOT_EXIST = "141";
    public static final String INVALID_REDIRECT_INFO = "142";
    public static final String INVALID_INDEXFILE_DETAIL = "143";
    public static final String CREATE_TOPIC_ERROR = "144";
    public static final String CYCLE_ROUTE_ERROR = "145";
    public static final String CONSUME_MSG_TIMEOUT = "146";
    public static final String SESSION_INIT_ERROR = "147";
    public static final String TRANSFER_QUEUE_REDIRECT = "148";


    public static final String PTP_SUCCESS = "E0000000000";
    public static final String PTP_SYSTEM_ERROR = "E0000000500";
    public static final String PTP_INVALID_REQUEST = "E0000000400";
    public static final String PTP_TIME_OUT = "E0000000601";

//    E0000000000	请求成功
//    E0000000404	请求资源不存在
//    E0000000500	系统异常
//    E0000000503	循环请求服务不可达
//    E0000000400	请求非法
//    E0000000403	请求资源未被授权
//    E0000000520	未知异常
//    E0000000600	系统不兼容
//    E0000000601	请求超时
//    E0000000602	无服务实例
//    E0000000603	数字证书校验异常
//    E0000000604	节点授权码已过期
//    E0000000605	节点组网时间已过期
//    E0000000606	对方节点已禁用网络
//    E0000000607	网络不通
//    E0000000614	接口未被许可调用
//    E0000000615	证书签名非法
//    E0000000616	报文编解码异常
//    E0000000617	下游版本不匹配服务不存在
//    E0000000618	节点或机构未组网
//    E0000000619	地址非法或无法访问
//    E0000000700	消息缓冲区满
//    E0000000608	会话已释放
//    E0000000413	报文超长BodyTooLarge


}
