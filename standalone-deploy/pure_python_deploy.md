You can deploy a standalone version FATE with source code according to the following steps. 

If you have met any problems during the installation process, please check Section 9 to check out if there exist any help. 

1. Check whether the local 8080,9360,9380 port is occupied.

   ```
   netstat -apln|grep 8080
   netstat -apln|grep 9360
   netstat -apln|grep 9380
   ```
   
2. Download source code from [github](https://github.com/FederatedAI/FATE)

3. Please install python with version 3.6 or 3.7. Then create a virtual environment:
```
cd(or create) {a dir you wish to locate your venv}
python -m venv {your venv_name}
cd {venv_name}    // This is your venv root 
source {venv root}/bin/activate
```

Alternatively, you can use conda to create a new environment.
```
cd(or create) {a dir you wish to locate your venv}
conda create -p ${your venv path} python=3.6     // or 3.7
conda activate ${your venv path}
```

4. Install all the requirements
```
cd {your fate root}/python
pip install -U pip
pip install -r requirements.txt
```

5. Configure the environment.  
```
cd {your fate root}
vi bin/init_env.sh
```
Config your PYTHONPATH and venv in this configuration file. 
```
export PYTHONPATH={your fate root}:{your fate root}/python
export EGGROLL_HOME=
venv={your venv root}        // venv is your virtual environment root
export JAVA_HOME=
export PATH=$PATH:$JAVA_HOME/bin
source ${venv}/bin/activate    // The way start your venv. If you are using conda, you may use conda activate ${venv} 
```

6. Config service conf
```
cd {your fate root}
vim conf/service_conf.yaml
```
The work mode are supposed to be 0 for standalone mode
```
work_mode: 0     // 0 for standalone mode, 1 for cluster mode.
use_registry: false
use_deserialize_safe_module: false
```

7. Start fate-flow server
```
cd {fate_root}/python/fate_flow
sh service.sh start
```

8. Test

   - Unit Test

```
cd {fate_root}
source bin/init_env.sh
bash ./python/federatedml/test/run_test.sh
```

   If success,  the screen shows like blow:

   ```
   there are 0 failed test
   ```

   - Toy_example Test

   ```
   cd {fate_root}
   source bin/init_env.sh
   python ./examples/toy_example/run_toy_example.py 10000 10000 0
   ```

   If success,  the screen shows like blow:

   ```
   success to calculate secure_sum, it is 2000.0
   ```

9. Some common problem
    * If you have occurs "Too many open files" error, you may need to change the limit of open files set by your system. For mac system, you can try [this](https://superuser.com/questions/433746/is-there-a-fix-for-the-too-many-open-files-in-system-error-on-os-x-10-7-1). For Linux system, try [this](http://woshub.com/too-many-open-files-error-linux/)
    * If gypm2 can not be installed, we recommend you install it with conda. Alternatively, for mac,you can try install the following basic dependencies by:
    ```
   brew install gmp mpfr libmpc
   ```

10. Install FATE-Client and FATE-Test

   To conveniently interact with FATE, we provide tools [FATE-Client](../python/fate_client) and [FATE-Test](../python/fate_test).

   Install FATE-Client and FATE-Test with the following commands:

   ```
   python -m pip install fate-client
   python -m pip install fate-test
   ```


Please refer [here](../examples/pipeline/README.rst) for a quick start tutorial.

You can also experience the fateboard access via a browser:
Http://hostip:8080.


Congratulations, You are all set. 

You can enjoy your FATE by following this [tutorial](../examples)