
# How to use gitee

[TOC]

## 1. Instructions

Try Gitee when you can't connect to Github for code

Note that you can only update code, but not push code, make issue by using Gitee, Please use Github

## 2. Start

```bash
git clone https://gitee.com/FederatedAI/FATE.git -b ${branch}
cd FATE
bash build/common/update_submodule_from_gitee.sh
```

Please set the **branch**.

## 3. Update

```bash
cd FATE
git pull
git submodule update --remote
```
