# FATE权限相关操作接口文档

### 1. 概述

用于fate本地鉴权,包含role、command、component、dataset等权限。

### 2. role、command、component权限操作接口
#### 2.1 授权
**简要描述：** 
- role授权
- command授权
- component授权

**请求URL：** 
- ` http://ip:port/v1/permission/grant/privilege`
  

**请求方式：**
- POST 
- content-application/json

**请求参数：** 

| 参数名          | 必选 | 类型   | 说明                                                     |
| :-------------- | :--- | :----- | -------------------------------------------------------- |
| privilege_role       | 否   | string | 赋予作为某种角色的权限, 取值为："guest", "host", "arbiter","all";其中"all"代表该类型全部权限都给，下同 。|
| privilege_command | 否   | string | 赋予请求方执行某个命令的权限 ,取值为: "create","stop","run","all"   |
| privilege_component   | 否   |  string   | 赋予请求方运行某个组件的权限, 取值为:所有算法组件类名小写,如:"dataio","heteronn"，"intersection",......,"all"|    

**请求参数示例**
```
{
    "privilege_role": "all",
    "privilege_command": "all",
    "privilege_component": "all"
}

```

**返回参数说明** 

| 参数名 | 类型   | 说明                     |
| :----- | :----- | ------------------------ |
| retcode | int    | 0表示成功，非0表示错误码 |
| retmsg    | string | 简要错误描述             |


#### 2.2 取消授权
**简要描述：** 
- 取消role授权
- 取消command授权
- 取消component授权
**请求URL：** 
- ` http://ip:port/v1/permission/delete/privilege`

  

**请求方式：**
- POST 
- content-application/json

**请求参数：** 

| 参数名          | 必选 | 类型   | 说明                                                     |
| :-------------- | :--- | :----- | -------------------------------------------------------- |
| privilege_role       | 否   | string | 赋予作为某种角色的权限, 取值为："guest", "host", "arbiter","all";其中"all"代表该类型全部权限都给，下同 。|
| privilege_command | 否   | string | 赋予请求方执行某个命令的权限 ,取值为: "create","stop","run","all"   |
| privilege_component   | 否   |  string   | 赋予请求方运行某个组件的权限, 取值为:所有算法组件类名小写,如:"dataio","heteronn"，"intersection",......,"all"|    


**请求参数示例**
```
{
    "privilege_role": "all",
    "privilege_command": "all",
    "privilege_component": "all"
}

```

**返回参数说明** 

| 参数名 | 类型   | 说明                     |
| :----- | :----- | ------------------------ |
| retcode | int    | 0表示成功，非0表示错误码 |
| retmsg    | string | 简要错误描述             |

#### 2.3 授权查询
**请求URL：** 
- ` http://ip:port/v1/permission/query/privilege`
  
**请求方式：**
- POST 
- content-application/json

**请求参数：** 

| 参数名          | 必选 | 类型   | 说明                                                     |
| :-------------- | :--- | :----- | -------------------------------------------------------- |
| src_party_id         | 是   | string | 表示请求方partyid |
| src_role    | 是   | string | 表示请求方角色, 如:"guest", "host", "arbiter |
| privilege_type|否   | string | 表示需要查询类型, 包含:"roles", "command", "component", 该参数不传或者传"all"即返回全部类型的全部查询结果|



**请求参数示例**
```
{
    "src_party_id": "9999",
    "src_role": "guest",
    "privilege_type": "all",
}

```

**返回参数说明** 

| 参数名 | 类型   | 说明                     |
| :----- | :----- | ------------------------ |
| retcode | int    | 0表示成功，非0表示错误码 |
| retmsg    | string | 简要错误描述             |
| data    | object | 权限查询结果    |
**返回示例*:*
```
{
    "data": {
        "privilege_command": ["create", "stop", "run"],
        "privilege_component": ["sbtfeaturetransformer", "heterodatasplit", "dataio"],
        "privilege_role": ["host", "arbiter", "guest"]
    },
    "retcode": 0,
    "retmsg": "success"
}
```

### 3. dataset权限操作接口
#### 3.1 授权
**简要描述：** 
- dataset授权
**请求URL：** 
- ` http://ip:port/v1/permission/grant/privilege`

**请求方式：**
- POST 
- content-application/json

**请求参数：** 

| 参数名          | 必选 | 类型   | 说明                                                     |
| :-------------- | :--- | :----- | -------------------------------------------------------- |
| src_user | 是   | string | 用数方用户名  |
| dest_user | 是   | string |供数方用户名  |
| privilege_dataset   | 是   | object    | 赋予用数方使用供数方某份或多份数据集的权限,参数如下|      

 **privilege dataset objec说明** 

| 参数名 | 类型   | 说明                     |
| :----- | :----- | ------------------------ |
| table_name | string    | 数据表表名 |
| namespace    | string | 数据表命名空间     |
**请求参数示例**
```
{
    "src_user": "user_a",
    "dest_user": "user_b",
    "privilege_dataset": [{"table_name": "xxx1", "namespace": "xxx1"}, {"table_name": "xxx2", "namespace": "xxx2"}]
}

```
携带权限过期时间:
```
{
    "src_user": "user_a",
    "dest_user": "user_b",
    "privilege_dataset": [[{"table_name": "xxx1", "namespace": "xxx1"}, 30*24*60*60]]
}
```

**返回参数说明** 

| 参数名 | 类型   | 说明                     |
| :----- | :----- | ------------------------ |
| retcode | int    | 0表示成功，非0表示错误码 |
| retmsg    | string | 简要错误描述             |

#### 3.2 取消授权
**简要描述：** 
- 取消dataset授权

**请求URL：** 
- ` http://ip:port/v1/permission/delete/privilege`
  

**请求方式：**
- POST 
- content-application/json

**请求参数：** 

| 参数名          | 必选 | 类型   | 说明                                                     |
| :-------------- | :--- | :----- | -------------------------------------------------------- |
| src_user | 是   | string | 用数方用户名  |
| dest_user | 是   | string |供数方用户名  |
| privilege_dataset   | 是   | object    | 取消用数方使用供数方某份数据集的权限,参数如下|          

 **privilege dataset objec说明** 

| 参数名 | 类型   | 说明                     |
| :----- | :----- | ------------------------ |
| table_name | string    | 数据表表名 |
| namespace    | string | 数据表命名空间     |
**请求参数示例**
```
{
    "src_user": "user_a",
    "dest_user": "user_b",
    "privilege_dataset": [{"table_name": "xxx1", "namespace": "xxx1"}, {"table_name": "xxx2", "namespace": "xxx2"}]
}

```

**返回参数说明** 

| 参数名 | 类型   | 说明                     |
| :----- | :----- | ------------------------ |
| retcode | int    | 0表示成功，非0表示错误码 |
| retmsg    | string | 简要错误描述             |
#### 3.3 授权查询
**请求URL：** 
- ` http://ip:port/v1/permission/query/privilege`
  

**请求方式：**
- POST 
- content-application/json

**请求参数：** 

| 参数名          | 必选 | 类型   | 说明                                                     |
| :-------------- | :--- | :----- | -------------------------------------------------------- |
| src_user | 是   | string | 用数方用户名  |
| dest_user | 是   | string |供数方用户名  |     


**请求参数示例**
```
{
    "src_user": "user_a",
    "dest_user": "user_b"
}
```


**返回参数说明** 

| 参数名 | 类型   | 说明                     |
| :----- | :----- | ------------------------ |
| retcode | int    | 0表示成功，非0表示错误码 |
| retmsg    | string | 简要错误描述             |
| data    | object | 权限查询结果    |



**返回示例** 
```
{
    "data": {
        "privilege_dataset": [{"table_name": "x1", "namespace": "x1"}, {"table_name": "x2", "namespace": "x2"}]
    },
    "retcode": 0,
    "retmsg": "success"
}
```