# 如何使用gitee

[TOC]

## 1. 说明

当你无法连接到Github获取代码时，可以尝试Gitee。

请注意，使用Gitee只能更新代码，而不能推送代码、发布问题，请使用Github。

## 2. 开始

```bash
git clone https://gitee.com/FederatedAI/FATE.git -b ${branch}
cd FATE
bash build/common/update_submodule_from_gitee.sh
```

请设置**branch**。

## 3. 更新

```bash
cd FATE
git pull
git submodule update --remote
```
