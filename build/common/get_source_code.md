# Get the source code

## 1. Setting Temporary Environment Variables

Set temporary environment variables (note that the environment variables set in the following way are only valid for the current terminal session, if you open a new terminal session, such as a new login or a new window, please set them again)

```bash
export branch={branch name, or using v and version number if you are using a release branch, e.g. v1.7.0}
export version={FATE version number, e.g. 1.7.0}
export version_tag={FATE version tag, e.g. rc1/rc2/release}
```

Example:

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

## 2. Get the code from Github

```bash
git clone https://github.com/FederatedAI/FATE.git -b ${branch} --recurse-submodules --depth=1
```

The **depth** parameter indicates that only the latest committed code is fetched, which can speed up cloning.

## 3. Get code from Gitee (try Gitee when you can't connect to Github to get code)

When you can't connect to Github to get your code, try Gitee.

Please note that with Gitee you can only update code, not push code or post issues, use Github.

```bash
git clone https://gitee.com/FederatedAI/FATE.git -b ${branch} --depth=1;
cd FATE;
bash build/common/update_submodule_from_gitee.sh
```

## 4. Update the code

```bash
cd FATE
git pull
git submodule update --remote
```
