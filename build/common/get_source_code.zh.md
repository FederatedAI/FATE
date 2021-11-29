# 获取源代码

## 1. 设置临时环境变量

设置临时环境变量(注意, 通过以下方式设置的环境变量仅在当前终端会话有效, 若打开新的终端会话, 如重新登录或者新窗口, 请重新设置)

```bash
export branch={分支名称, 若使用某个发布版本分支, 则为`v版本号`, 如`v1.7.0`}
export version={FATE版本号, 如1.7.0}
export version_tag={FATE版本标签, 如rc1/rc2/release}
```

样例:

```bash
export branch=v1.7.0
export version=1.7.0
export version_tag=release
```

```bash
export branch=develop-1.7
export version=1.7.0
export version_tag=release
```

## 2. 从Github获取代码

```bash
git clone https://github.com/FederatedAI/FATE.git -b ${branch} --recurse-submodules --depth=1
```

**depth**参数表示只获取最新提交的代码，这可以加快克隆的速度。

## 3. 从Gitee获取代码（当你无法连接到Github获取代码时，可以试试Gitee）

当你无法连接到Github获取代码时，可以尝试Gitee。

请注意，使用Gitee只能更新代码，而不能推送代码、发布问题，请使用Github。

```bash
git clone https://gitee.com/FederatedAI/FATE.git -b ${branch} --depth=1;
cd FATE;
bash build/common/update_submodule_from_gitee.sh
```

## 4. 更新代码

```bash
cd FATE
git pull
git submodule update --remote
```
