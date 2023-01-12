# 一、背景：

FATE1.X维护了多套通信架构，包括eggroll、spark+pulsar+nginx 、spark+rabbitmq+nginx。

##### 下图为FATE1.X采用eggroll为计算引擎时的通信架构

![fate_old_eggroll](../images/fate_old_eggroll.png)

##### 下图为FATE1.X采用spark为计算引擎时的通信架构

![fate_old_spark](../images/fate_old_spark.png)

如上所示，FATE1.X通信架构有一些弊端如下：

- 需要维护多套通信组件来支持不同的计算引擎
- 多种计算引擎组网困难，难以统一路由以及流控
- eggroll通信只支持同步rpc调用+流式传输，不支持消息队列类型的异步传输
- pulsar 以及rabbitmq等集群不易安装以及维护，程序不容易感知集群间传输时出现的网络异常
- 不容易对消息队列组件进行自定义开发等

为了解决以上问题，我们预备在FATE2.X中使用统一的通信组件OSX，统一支持不同计算引擎

# 二、新组件OSX:



## 新组件特性：

- 站点之间通信同时支持grpc、http1.X协议
- 支持多种计算引擎传输，包括eggroll、spark
- 支持在FATE1.X中无缝替代rollsite组件
- 同时支持同步rpc调用+消息队列
- 支持作为exchange中心节点部署 ，支持FATE1.X  、FATE2.X 接入
- 支持集群流量控制，可以针对不同参与方制定流控策略
- 路由配置与原eggroll基本一致，降低了移植难度
- 支持集群模式与standalone两种模式（默认为standalone模式，standalone已可满足大部分场景）
- 可根据接口中厂商编码可针对不同技术提供商做自定义开发

## 组件设计：



## ![osx_component](../images/osx_component.png)

## 部署架构：

采用eggroll作为计算引擎时的部署架构:

![osx_on_eggroll](../images/osx_on_eggroll.png)

上图为采用spark作为计算引擎时的部署架构:

![osx_on_spark.drawio](../images/osx_on_spark.drawio.png)



## 接口：

```protobuf
message Message{
  string msgId = 1;//消息ID
  bytes head = 2;//消息头部
  bytes body = 3;//消息体
}
message  TopicInfo{
  string topic=1;
  string ip = 2;
  int32  port = 3;
  int64  createTimestamp = 4;
  int32  status = 5;
}

// PTP Private transfer protocol
// 通用报头名称编码，4层无Header以二进制填充到报头，7层以Header传输
enum Header {
  Version = 0;           // 协议版本               对应7层协议头x-ptp-version
  TechProviderCode = 1;  // 厂商编码               对应7层协议头x-ptp-tech-provider-code
  TraceID = 4;           // 链路追踪ID             对应7层协议头x-ptp-trace-id
  Token = 5;             // 认证令牌               对应7层协议头x-ptp-token
  SourceNodeID = 6;      // 发送端节点编号          对应7层协议头x-ptp-source-node-id
  TargetNodeID = 7;      // 接收端节点编号          对应7层协议头x-ptp-target-node-id
  SourceInstID = 8;      // 发送端机构编号          对应7层协议头x-ptp-source-inst-id
  TargetInstID = 9;      // 接收端机构编号          对应7层协议头x-ptp-target-inst-id
  SessionID = 10;        // 通信会话号，全网唯一     对应7层协议头x-ptp-session-id
}

// 通信扩展元数据编码，扩展信息均在metadata扩展
enum Metadata {
  MessageTopic = 0;                    // 消息话题，异步场景
  MessageCode = 1;                     // 消息编码，异步场景
  SourceComponentName = 2;             // 源组件名称
  TargetComponentName = 3;             // 目标组件名称
  TargetMethod = 4;                    // 目标方法
  MessageOffSet = 5;                   // 消息序列号
  InstanceId = 6;                      // 实例ID
  Timestamp  = 7;                      // 时间戳
}

// 通信传输层输入报文编码
message Inbound {
  map<string, string>  metadata = 1;   // 报头，可选，预留扩展，Dict，序列化协议由通信层统一实现
  bytes payload = 2;                   // 报文，上层通信内容承载，序列化协议由上层基于SPI可插拔
}

// 通信传输层输出报文编码
message Outbound {
  map<string, string>  metadata = 1;  // 报头，可选，预留扩展，Dict，序列化协议由通信层统一实现
  bytes payload = 2;                  // 报文，上层通信内容承载，序列化协议由上层基于SPI可插拔
  string code = 3;                    // 状态码
  string message = 4;                 // 状态说明
}

// 互联互通如果使用异步传输协议作为标准参考，Header会复用metadata传输互联互通协议报头，且metadata中会传输异步场景下的消息相关属性
// 互联互通如果使用其他协议作为参考标准，Header会复用metadata传输互联互通协议报头
// 互联互通如果使用GRPC作为参考标准，Header会复用HTTP2的报头传输互联互通协议报头

service PrivateTransferProtocol {
  rpc transport (stream Inbound) returns (stream Outbound);
  rpc invoke (Inbound) returns (Outbound);
}
```





