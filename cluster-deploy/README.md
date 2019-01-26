# Build
-

## 1. Checkout from git
```
git clone https://github.com/WeBankFinTech/FATE.git
```

## 2. Java Components

### 2.1. Compile
```
cd arch
maven clean package -DskipTests
```

### 2.2. Packaging
The next step is copying jar files to your destination directory.

There is a script in this project that can ease this. It packs up target jars and dependent libraries in one tar.gz file. You can find it under cluster-deploy/scripts/packaging.sh.

You need to change `base_dir` and `output_dir` with respect to the arch directory of this project and output directory. 

Then you can run the following command:

```
bash packaging.sh
```
If everything is ok, tar files can be found in `output_dir`. 

### 2.3. Configuration Files
Although configuration path is flexible, we recommend users to organize them well.
Example configuration files can be found under 
`arch/${module_name}/src/main/resources/`

Users can find a detailed configuration document in 
[`cluster-deploy/doc` ](https://https://github.com/WeBankFinTech/FATE/cluster-deploy/doc)

### 2.4. How to run
Main function is named after module name. Currently we have following Main functions:

Number | Module Name     | Main Function
-------|-----------------|---------------
1      | federation      | com.webank.ai.fate.driver.Federation
2      | meta-service    | com.webank.ai.fate.eggroll.MetaService
3      | proxy           | com.webank.ai.fate.networking.Proxy
4      | roll            | com.webank.ai.fate.eggroll.Roll
5      | storage-service | com.webank.ai.fate.eggroll.StorageService

Please note that users should add directory of configuration files to Java's classpath, so that these configurations can be loaded.

We provide example management scripts to run these services. Users can find them under [`cluster-deploy/example-dir-tree`](https://https://github.com/WeBankFinTech/FATE/cluster-deploy/example-dir-tree), along with example directory tree described in section 5.

## 3. Python Components


## 4. How to Run in Cluster Mode
Please refer to configuation guide [here](https://https://github.com/WeBankFinTech/FATE/cluster-deploy/doc/configuration.md)


## 5. Example Directory Tree
-
```
deploy-dir
|--- federation
|    |- conf/
|    |- lib/
|    |- fate-federation-0.1.jar
|    |- fate-federation.jar -> fate-fedaration-0.1.jar
|
|--- meta-service
|    |- conf/
|    |- lib/
|    |- fate-meta-service-0.1.jar
|    |- fate-mata-service.jar -> fate-meta-service-0.1.jar
|
|--- proxy
|    |- conf/
|    |- lib/
|    |- fate-proxy-0.1.jar
|    |- fate-proxy.jar -> fate-proxy-0.1.jar
|
|--- python
|    |- arch --- |- api/
|                |- conf/
|                |- processor/              
|    |- federatedml/
|    |- examples/
|    |- workflow/
|
|--- roll
|    |- conf/
|    |- lib/
|    |- fate-roll-0.1.jar
|    |- fate-roll.jar -> fate-roll-0.1.jar
|
|--- storage-service
|    |- conf/
|    |- lib/
|    |- fate-storage-service-0.1.jar
|    |- fate-storage-service.jar -> fate-storage-service-0.1.jar

```