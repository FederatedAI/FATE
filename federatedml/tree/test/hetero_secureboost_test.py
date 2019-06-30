#
#  Copyright 2019 Toe FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import time
import os
import subprocess
import threading


def set_jobid(path, jobid):
    fin = open(path, "r")
    with open(path + "_bak", "w") as fout:
        for line in fin:
            if line.find("jobid") == -1:
                fout.write(line)
            else:
                fout.write(line.replace("jobid", jobid))


def run_secureboost_host_test(jobid):
    set_jobid("./hetero_secure_boost_host_test.py", jobid)
    cmd = ["python", "./hetero_secure_boost_host_test.py_bak"]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    stderr = stderr.decode("utf-8")
    print (stderr)
    process.wait()
    os.system("rm hetero_secure_boost_host_test.py_bak")


def run_secureboost_guest_test(jobid):
    set_jobid("./hetero_secure_boost_guest_test.py", jobid)
    cmd = ["python", "./hetero_secure_boost_guest_test.py_bak"]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    stderr = stderr.decode("utf-8")
    print (stderr)
    process.wait()
    os.system("rm hetero_secure_boost_guest_test.py_bak")


if __name__ == "__main__":
    jobid = "test_hetero_secureboost_" + str(int(time.time() * 1000))

    host_job = threading.Thread(target=run_secureboost_host_test, args=[jobid])
    
    guest_job = threading.Thread(target=run_secureboost_guest_test, args=[jobid])

    host_job.start()
    guest_job.start()

    host_job.join()
    guest_job.join()
