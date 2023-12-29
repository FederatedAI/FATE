/*
 * Copyright 2019 The FATE Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.fedai.osx.core.utils;

import org.apache.commons.lang3.StringUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.lang.management.GarbageCollectorMXBean;
import java.lang.management.ManagementFactory;
import java.net.Inet6Address;
import java.net.InetAddress;
import java.net.NetworkInterface;
import java.net.UnknownHostException;
import java.util.Enumeration;
import java.util.List;
import java.util.Optional;
import java.util.regex.Pattern;

public class GetSystemInfo {

    private static final Logger logger = LoggerFactory.getLogger(GetSystemInfo.class);
    private static final Pattern ADDRESS_PATTERN = Pattern.compile("^\\d{1,3}(\\.\\d{1,3}){3}\\:\\d{1,5}$");
    private static final Pattern LOCAL_IP_PATTERN = Pattern.compile("127(\\.\\d{1,3}){3}$");
    private static final Pattern IP_PATTERN = Pattern.compile("\\d{1,3}(\\.\\d{1,3}){3,5}$");
    private static final String SPLIT_IPV4_CHARECTER = "\\.";
    private static final String SPLIT_IPV6_CHARECTER = ":";
    public static String localIp;
    private static String ANYHOST_KEY = "anyhost";
    private static String ANYHOST_VALUE = "0.0.0.0";
    private static String LOCALHOST_KEY = "localhost";
    private static String LOCALHOST_VALUE = "127.0.0.1";

    static {
        localIp = getLocalIp();
        logger.info("set local ip : {}", localIp);
    }

    public static int getPid() {
        String name = ManagementFactory.getRuntimeMXBean().getName();
        return Integer.parseInt(name.split("@")[0]);
    }

    public static String getLocalIp() {

        try {
            InetAddress inetAddress = getLocalAddress0("eth0");
            if (inetAddress != null) {
                return inetAddress.getHostAddress();
            } else {
                inetAddress = getLocalAddress0("");
            }
            if (inetAddress != null) {
                return inetAddress.getHostAddress();
            } else {
                throw new RuntimeException("can not get local ip");
            }

        } catch (Throwable e) {
            logger.error(e.getMessage(), e);
        }
        return "";
    }

    private static InetAddress getLocalAddress0(String name) {
        InetAddress localAddress = null;
        try {
            localAddress = InetAddress.getLocalHost();
            Optional<InetAddress> addressOp = toValidAddress(localAddress);
            if (addressOp.isPresent()) {
                return addressOp.get();
            } else {
                localAddress = null;
            }
        } catch (Throwable e) {
            logger.warn(e.getMessage());
        }

        try {
            Enumeration<NetworkInterface> interfaces = NetworkInterface.getNetworkInterfaces();
            if (null == interfaces) {
                return localAddress;
            }
            while (interfaces.hasMoreElements()) {
                try {
                    NetworkInterface network = interfaces.nextElement();
                    if (network.isLoopback() || network.isVirtual() || !network.isUp()) {
                        continue;
                    }
                    if (StringUtils.isNotEmpty(name)) {
                        if (!network.getName().equals(name)) {
                            continue;
                        }
                    }
                    Enumeration<InetAddress> addresses = network.getInetAddresses();
                    while (addresses.hasMoreElements()) {
                        try {
                            Optional<InetAddress> addressOp = toValidAddress(addresses.nextElement());
                            if (addressOp.isPresent()) {
                                try {
                                    if (addressOp.get().isReachable(10000)) {
                                        return addressOp.get();
                                    }
                                } catch (IOException e) {
                                    // ignore
                                }
                            }
                        } catch (Throwable e) {
                            logger.warn(e.getMessage());
                        }
                    }
                } catch (Throwable e) {
                    logger.warn(e.getMessage());
                }
            }
        } catch (Throwable e) {
            logger.warn(e.getMessage());
        }
        return localAddress;
    }

    public static String getOsName() {

        String osName = System.getProperty("os.name");
        return osName;
    }

    static boolean isPreferIpv6Address() {
        boolean preferIpv6 = Boolean.getBoolean("java.net.preferIPv6Addresses");
        if (!preferIpv6) {
            return false;
        }
        return false;
    }

    static InetAddress normalizeV6Address(Inet6Address address) {
        String addr = address.getHostAddress();
        int i = addr.lastIndexOf('%');
        if (i > 0) {
            try {
                return InetAddress.getByName(addr.substring(0, i) + '%' + address.getScopeId());
            } catch (UnknownHostException e) {
                // ignore
                logger.debug("Unknown IPV6 address: ", e);
            }
        }
        return address;
    }

    private static Optional<InetAddress> toValidAddress(InetAddress address) {
        if (address instanceof Inet6Address) {
            Inet6Address v6Address = (Inet6Address) address;
            if (isPreferIpv6Address()) {
                return Optional.ofNullable(normalizeV6Address(v6Address));
            }
        }
        if (isValidV4Address(address)) {
            return Optional.of(address);
        }
        return Optional.empty();
    }

    static boolean isValidV4Address(InetAddress address) {
        if (address == null || address.isLoopbackAddress()) {
            return false;
        }
        String name = address.getHostAddress();
        boolean result = (name != null
                && IP_PATTERN.matcher(name).matches()
                && !ANYHOST_VALUE.equals(name)
                && !LOCALHOST_VALUE.equals(name));
        return result;
    }


//    public static double getSystemCpuLoad() {
//        OperatingSystemMXBean osmxb = (OperatingSystemMXBean) ManagementFactoryHelper
//                .getOperatingSystemMXBean();
//        double systemCpuLoad = osmxb.getSystemCpuLoad();
//        return systemCpuLoad;
//    }


//    public static double getProcessCpuLoad() {
//        OperatingSystemMXBean osmxb = (OperatingSystemMXBean) ManagementFactoryHelper
//                .getOperatingSystemMXBean();
//        double processCpuLoad = osmxb.getProcessCpuLoad();
//        return processCpuLoad;
//    }


//    public static long getTotalMemorySize() {
//        int kb = 1024;
//        OperatingSystemMXBean osmxb = (OperatingSystemMXBean) ManagementFactoryHelper
//                .getOperatingSystemMXBean();
//
//
//        long totalMemorySize = osmxb.getTotalPhysicalMemorySize() / kb;
//        return totalMemorySize;
//    }


//    public static long getFreePhysicalMemorySize() {
//        int kb = 1024;
//        OperatingSystemMXBean osmxb = (OperatingSystemMXBean) ManagementFactory
//                .getOperatingSystemMXBean();
//        long freePhysicalMemorySize = osmxb.getFreePhysicalMemorySize() / kb;
//        return freePhysicalMemorySize;
//    }


//    public static long getUsedMemory() {
////        int kb = 1024;
////        OperatingSystemMXBean osmxb = (OperatingSystemMXBean) ManagementFactory
////                .getOperatingSystemMXBean();
////        long usedMemory = (osmxb.getTotalPhysicalMemorySize() - osmxb.getFreePhysicalMemorySize()) / kb;
////        return usedMemory;
////    }


    public static void getJVMRuntimeParam() {

//        RuntimeMXBean runtimeMXBean = ManagementFactoryHelper.getRuntimeMXBean();
//JVM启动参数
//        System.out.println(runtimeMXBean.getInputArguments());
//系统属性
//        System.out.println(runtimeMXBean.getSystemProperties());
//JVM名字
//        System.out.println(runtimeMXBean.getVmName());


    }


//    public  static void  getJvmThreadInfo(){
//        java.lang.management.ThreadMXBean threadMXBean =  ManagementFactoryHelper.getThreadMXBean();
//        threadMXBean.getThreadCount();
//        threadMXBean.getCurrentThreadCpuTime();
//        threadMXBean.getCurrentThreadUserTime();
//    }


    private static void reportGC() {
        long fullCount = 0, fullTime = 0, youngCount = 0, youngTime = 0;
        List<GarbageCollectorMXBean> gcs = ManagementFactory.getGarbageCollectorMXBeans();
        for (GarbageCollectorMXBean gc : gcs) {
            System.err.println(gc.getName() + "" + gc.getCollectionCount());


        }
//            switch (GarbageCollectorName.of(gc.getName())) {
//                case MarkSweepCompact:
//                case PSMarkSweep:
//                case ConcurrentMarkSweep:
//                    fullCount += gc.getCollectionCount();
//                    fullTime += gc.getCollectionTime();
//                    break;
//                case Copy:
//                case ParNew:
//                case PSScavenge:
//                    youngCount += gc.getCollectionCount();
//                    youngTime += gc.getCollectionTime();
//                    break;
//            }
        //todo your deal code， perfcounter report or write log here
    }


//    public  static  void  getSystemLoad(){
//
//        java.lang.management.OperatingSystemMXBean operatingSystemMXBean = ManagementFactoryHelper.getOperatingSystemMXBean();
////获取服务器的CPU个数
////          System.out.println(operatingSystemMXBean.());
////获取服务器的平均负载。这个指标非常重要，它可以有效的说明当前机器的性能是否正常，如果load过高，说明CPU无法及时处理任务。
////        System.out.println(operatingSystemMXBean.getSystemLoadAverage());
//
//
//
//    }


    public static void main(String[] args) {

        //    getSystemLoad();
        reportGC();


    }


}

