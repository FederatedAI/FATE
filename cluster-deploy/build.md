

#                      **FATE Cluster Build Guide**

## 1. Cluster Architecture

## 2. Software Environment

| Name         | Version | 
| -------------| --------|
| JDK          | 1.8+    |
| Maven        | 3.6.1+  |

## 3. Build
```bash
git clone https://github.com/FederatedAI/FATE.git -b v1.5.0 --depth=1
cd FATE/cluster-deploy
bash build.sh release all
```
The **depth** parameter represents the code that only gets the latest commit, which can speed up the clone.
The **release** is the version tag, which you can modify.
The **all** means that modules needs to be deployed, all means all, supports all, python, fateboard, eggroll

## 4. Output
```bash
ls -l FATE_install_1.5.0-release.tar.gz
```

## 5. Check packages
```bash
tar xzf FATE_install_1.5.0-release.tar.gz
ls -lrt FATE_install_1.5.0
```
You can see the following package:
- bin.tar.gz
- conf.tar.gz
- eggroll.tar.gz
- examples.tar.gz
- fateboard.tar.gz
- fate.env
- packages_md5.txt
- python.tar.gz
- RELEASE.md
- requirements.txt

| Name         | Details| 
| -------------| --------|
| bin.tar.gz   | some scripts|
| conf.tar.gz   | some configuration files|
| eggroll.tar.gz | eggroll cluster: cluster manager, node manager, rollsiter  |
| examples.tar.gz | some algorithm test examples|
| fateboard.tar.gz | fateboard packages|
| fate.env | settings for version|
| packages_md5.txt | md5 numbers for each package|
| python.tar.gz | include federatedml and fate flow|
| RELEASE.md | release document|
| requirements.txt | necessary dependency for python environment|
