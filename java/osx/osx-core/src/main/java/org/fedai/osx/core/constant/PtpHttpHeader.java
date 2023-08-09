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
package org.fedai.osx.core.constant;

public class PtpHttpHeader {

//    Version = 0;           // 协议版本               对应7层协议头x-ptp-version
//    TechProviderCode = 1;  // 厂商编码               对应7层协议头x-ptp-tech-provider-code
//    TraceID = 4;           // 链路追踪ID             对应7层协议头x-ptp-trace-id
//    Token = 5;             // 认证令牌               对应7层协议头x-ptp-token
//    SourceNodeID = 6;      // 发送端节点编号          对应7层协议头x-ptp-source-node-id
//    TargetNodeID = 7;      // 接收端节点编号          对应7层协议头x-ptp-target-node-id
//    SourceInstID = 8;      // 发送端机构编号          对应7层协议头x-ptp-source-inst-id
//    TargetInstID = 9;      // 接收端机构编号          对应7层协议头x-ptp-target-inst-id
//    SessionID = 10;        // 通信会话号，全网唯一     对应7层协议头x-ptp-session-id

//    MessageTopic = 0;                    // 消息话题，异步场景
//    MessageCode = 1;                     // 消息编码，异步场景
//    SourceComponentName = 2;             // 源组件名称
//    TargetComponentName = 3;             // 目标组件名称
//    TargetMethod = 4;                    // 目标方法
//    MessageOffSet = 5;                   // 消息序列号
//    InstanceId = 6;                      // 实例ID
//    Timestamp  = 7;                      // 时间戳

    static public  final String  Version="x-ptp-version";
    static public  final String  TechProviderCode  = "x-ptp-tech-provider-code";
    static public  final String   TraceID = "x-ptp-trace-id";
    static public  final String   Token = "x-ptp-token";
    static public  final String   SourceNodeID = "x-ptp-source-node-id";
    static public  final String   TargetNodeID = "x-ptp-target-node-id";
    static public  final String   SourceInstID = "x-ptp-source-inst-id";
    static public  final String   TargetInstID = "x-ptp-target-inst-id";
    static public  final String   SessionID = "x-ptp-session-id";
    static public  final String   MessageTopic = "x-ptp-message-topic";
    static public  final String   MessageCode = "x-ptp-message-code";
    static public  final String   SourceComponentName = "x-ptp-source-component-name";
    static public  final String   TargetComponentName = "x-ptp-target-component-name";

    static public final String   TargetMethod = "x-ptp-target-method";
    static public final String   SourceMethod = "x-ptp-source-method";

    static public final String   MessageOffSet = "x-ptp-message-offset";
    static public final String   InstanceId = "x-ptp-instance-id";
    static public final String   Timestamp = "x-ptp-timestamp";

    static public final String  MessageFlag = "x-ptp-message-flag";
    static public final String   ReturnCode = "x-ptp-code";
    static public final String   ReturnMessage = "x-ptp-message";
    static public final String JobId = "x-ptp-job-id";

}
