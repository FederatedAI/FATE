# Build

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

`base_dir` is the path of `FATE/arch` in your local environment, and `output_dir` is the dir where you want to put the packging output tar files.

Then you can run the following command:

```
bash packaging.sh
```
If everything is ok, tar files can be found in `output_dir`. 

Output tar file names are like `fate-${module}-${version}.tar.gz`. It contains `fate-${module}-${version}.jar` and a `lib/` dir which contains dependent libraries for the corresponding `fate-${module}-${version}.jar`. 

[`cluster-deploy/example-dir-tree`](https://github.com/WeBankFinTech/FATE/tree/master/cluster-deploy/example-dir-tree) contains an example dir trees. You can extract each tar file in the corresponding dir.

Let's take `federation` module as an example. After packaing for version 0.1, you get a `fate-federation-0.1.tar.gz`. Then you can perform the followings:

1. `cd ${path-to-example-dir-tree}/federation`. 
2. `tar xzf fate-federation-0.1.tar.gz`, so that tar file is extracted.
3. `cp -r FATE/arch/driver/federation/src/main/resources/ conf/`, so that example configuration files are copied and a `conf` dir is created for them.
4. Modify configuration files. See section 2.3 for more details.
5. `ln -s fate-federation-0.1.jar fate-federation.jar` to create a symlink, removing version-specific dependency for other tools.

You can perform the same to Java modules.

### 2.3. Configuration Files
Although configuration path is flexible, we recommend users to organize them well.
Example configuration files can be found under 
`arch/${module_name}/src/main/resources/`

Users can find a detailed configuration document in 
[`cluster-deploy/doc` ](https://github.com/WeBankFinTech/FATE/tree/master/cluster-deploy/doc)

### 2.4. How to run
Main function is named after module name. Currently we have following Main functions:

Number | Module Name     | Main Function                              | Configuration Example
-------|-----------------|--------------------------------------------|-----------------------------
1      | federation      | com.webank.ai.fate.driver.Federation       | FATE/arch/driver/federation/src/main/resources/
2      | meta-service    | com.webank.ai.fate.eggroll.MetaService     | FATE/arch/eggroll/meta-service/src/main/resources/
3      | proxy           | com.webank.ai.fate.networking.Proxy        | FATE/arch/networking/proxy/src/main/resources/
4      | roll            | com.webank.ai.fate.eggroll.Roll            | FATE/arch/eggroll/roll/src/main/resources/
5      | storage-service | com.webank.ai.fate.eggroll.StorageService  | FATE/arch/eggroll/storage-service/src/main/resources/

Please note that users should add directory of configuration files to Java's classpath, so that these configurations can be loaded.

We have also provided example management scripts to run these services. Users can find them under [`cluster-deploy/example-dir-tree`](https://github.com/WeBankFinTech/FATE/tree/master/cluster-deploy/example-dir-tree), along with example directory tree described in section 5.

## 3. Python Components

### 3.1. Packaging

```
mkdir -p ${path-to-example-dir-tree}/python
git archive -o ${path-to-example-dir-tree}/python/python.tar $(git rev-parse HEAD) arch/api arch/processor federatedml workflow examples
cd ${path-to-example-dir-tree}/python
tar -xf python.tar
```

### 3.2. Configuration Files
Configuration file path: 
`python/arch/conf/server_conf.json`

Users can find a detailed configuration document in 
[`cluster-deploy/doc` ](https://github.com/WeBankFinTech/FATE/tree/master/cluster-deploy/doc)

### 3.3 How to run

```
#enter virtual env fisrt
(venv) $ pip install -r requirements.txt
(venv) $ export PYTHONPATH=${path-to-example-dir-tree}/python
# run processor
(venv) $ python ${path-to-example-dir-tree}/python/processor/processor.py 2>&1 > ${path-to-example-dir-tree}/python/processor.out &
# run task_manager
(venv) $ python ${path-to-example-dir-tree}/python/task_manager/manager.py 2>&1 > ${path-to-example-dir-tree}/python/manager.out &
```


## 4. How to Run in Cluster Mode
Please refer to configuation guide [here](https://github.com/WeBankFinTech/FATE/tree/master/cluster-deploy/doc/configuration.md)


## 5. Example Directory Tree

```
deploy-dir
|
|--- federation
|    |- conf/
|    |  |- applicationContext-federation.xml
|    |  |- federation.properties
|    |  |- log4j2.properties
|    |
|    |- lib/
|    |- fate-federation-0.1.jar
|    |- fate-federation.jar -> fate-fedaration-0.1.jar
|
|--- meta-service
|    |- conf/
|    |  |- applicationContext-meta-service.xml
|    |  |- jdbc.properties
|    |  |- log4j2.properties
|    |  |- meta-service.properties
|    |
|    |- lib/
|    |- fate-meta-service-0.1.jar
|    |- fate-mata-service.jar -> fate-meta-service-0.1.jar
|
|--- proxy
|    |- conf/
|    |  |- applicationContext-proxy.xml
|    |  |- log4j2.properties
|    |  |- proxy.properties
|    |  |- route_table.json
|    |
|    |- lib/
|    |- fate-proxy-0.1.jar
|    |- fate-proxy.jar -> fate-proxy-0.1.jar
|
|--- python
|    |- arch
|    |  |- api/
|    |  |- conf/
|    |  |- processor/
|    |  |- task_manager/
|    |
|    |- federatedml/
|    |- examples/
|    |- workflow/
|
|--- roll
|    |- conf/
|    |  |- applicationContext-roll.xml
|    |  |- log4j2.properties
|    |  |- roll.properties
|    |
|    |- lib/
|    |- fate-roll-0.1.jar
|    |- fate-roll.jar -> fate-roll-0.1.jar
|
|--- storage-service
|    |- conf/
|    |  |- log4j2.properties
|    |
|    |- lib/
|    |- fate-storage-service-0.1.jar
|    |- fate-storage-service.jar -> fate-storage-service-0.1.jar
|
|--- serving-server
|    |- conf/
|    |  |- log4j2.properties
|    |  |- serving-server.properties
|    |
|    |- lib/
|    |- fate-serving-server-0.1.jar
|    |- fate-serving-server.jar -> fate-serving-server-0.1.jar

```

## 6. Future works
Deploy and build will be automated in future releases.
