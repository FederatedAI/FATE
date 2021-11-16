RELEASE_VERSION_TAG_NAME="release"

kernel=`uname -s`
echo "[INFO] kernel: ${kernel}"
case "${kernel}" in
    "Darwin")
            OS=MacOS
            alias fate_sed_cmd="sed -i ''"
            ;;
    "Linux")
            system=`sed -e '/"/s/"//g' /etc/os-release | awk -F= '/^NAME/{print $2}'`
            echo "[INFO] linux system: ${system}"

            case "${system}" in
                "CentOS Linux")
                        OS=CentOS
                        ;;
                "Ubuntu")
                        OS=Ubuntu
                        ;;
                *)
                        echo "Not support this system."
                        exit -1
            esac

            alias fate_sed_cmd="sed -i"

            ;;
    *)
            echo "Not support this kernel."
            exit -1
esac

echo "[INFO] os: ${OS}"
