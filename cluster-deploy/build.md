

#                      **FATE Cluster Build Guide**

## 1. Cluster Architecture

## 2. Software Environment

| Name         | Version | 
| -------------| --------|
| JDK          | 1.8+    |
| Maven        | 3.6.1+  |

## 3. Build
```bash
git clone https://github.com/FederatedAI/FATE.git -b v1.4.3 --depth=1
cd FATE/cluster-deploy
bash build.sh release 
```
The **depth** parameter represents the code that only gets the latest commit, which can speed up the clone.
The **release** is the version tag, which you can modify.

## 4. Output
```bash
ls -l FATE_install_1.4.3-release.tar.gz
```

## 5. Check packages
```bash
tar xzf FATE_install_1.4.3-release.tar.gz
ls -lrt FATE_install_1.4.3
```
You can see the following package:
- bin
- python.tar.gz
- fateboard.tar.gz
- eggroll.tar.gz

| Name         | Details| 
| -------------| --------|
| bin          | some scripts|
| python.tar.gz | federatedml and fate flow|
| fateboard.tar.gz | fateboard |
| eggroll.tar.gz | eggroll cluster: cluster manager, node manager, rollsiter  |
