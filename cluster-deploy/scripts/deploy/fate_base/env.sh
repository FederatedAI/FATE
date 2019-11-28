#!/bin/bash
#sudo yum -y install gcc gcc-c++ make openssl-devel supervisor gmp-devel mpfr-devel libmpc-devel libaio numactl autoconf automake libtool libffi-devel snappy snappy-devel zlib zlib-devel bzip2 bzip2-devel lz4-devel libasan lsof

system=`sed -e '/"/s/"//g' /etc/os-release | awk -F= '/^NAME/{print $2}'`
echo ${system}
case "${system}" in
    "CentOS Linux")
            echo "CentOS System"
            sudo yum -y install gcc gcc-c++ make openssl-devel supervisor gmp-devel mpfr-devel libmpc-devel libaio numactl autoconf automake libtool libffi-devel snappy snappy-devel zlib zlib-devel bzip2 bzip2-devel lz4-devel libasan lsof
            ;;
    "Ubuntu")
            echo "Ubuntu System"
            sudo apt-get install -y gcc g++ make  openssl supervisor libgmp-dev  libmpfr-dev libmpc-dev libaio1 libaio-dev numactl autoconf automake libtool libffi-dev libssl1.0.0 libssl-dev  liblz4-1 liblz4-dev liblz4-1-dbg liblz4-tool  zlib1g zlib1g-dbg zlib1g-dev
            cd /usr/lib/x86_64-linux-gnu
            if [ ! -f "libssl.so.10" ];then
                 sudo ln -s libssl.so.1.0.0 libssl.so.10
                 sudo ln -s libcrypto.so.1.0.0 libcrypto.so.10
            fi
            ;;
    *)
            echo "Not support this system."
            exit -1
esac
