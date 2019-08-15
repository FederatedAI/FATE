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
package com.webank.ai.fate.board.utils;

import com.sun.management.OperatingSystemMXBean;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.management.ManagementFactory;
import java.lang.management.MemoryPoolMXBean;
import java.net.*;
import java.util.Enumeration;
import java.util.List;

public class GetSystemInfo {


    static Logger logger = LoggerFactory.getLogger(GetSystemInfo.class);


    public static String getLocalIp() {

        String sysType = System.getProperties().getProperty("os.name");
        String ip;
        if (sysType.toLowerCase().startsWith("win")) {
            String localIP = null;
            try {
                localIP = InetAddress.getLocalHost().getHostAddress();
            } catch (UnknownHostException e) {
                logger.error(e.getMessage(), e);
            }
            if (localIP != null) {
                return localIP;
            }
        } else {
            ip = getIpByEthNum("eth0");
            if (ip != null) {
                return ip;
            }
        }
        return "";
    }

    private static String getIpByEthNum(String ethNum) {
        try {
            Enumeration allNetInterfaces = NetworkInterface.getNetworkInterfaces();
            InetAddress ip;
            while (allNetInterfaces.hasMoreElements()) {
                NetworkInterface netInterface = (NetworkInterface) allNetInterfaces.nextElement();
                if (ethNum.equals(netInterface.getName())) {
                    Enumeration addresses = netInterface.getInetAddresses();
                    while (addresses.hasMoreElements()) {
                        ip = (InetAddress) addresses.nextElement();
                        if (ip != null && ip instanceof Inet4Address) {
                            return ip.getHostAddress();
                        }
                    }
                }
            }
        } catch (SocketException e) {
            logger.error(e.getMessage(), e);
        }
        return "";
    }


    public static String getOsName() {

        String osName = System.getProperty("os.name");
        return osName;
    }


    public static double getSystemCpuLoad() {
        OperatingSystemMXBean osmxb = (OperatingSystemMXBean) ManagementFactory
                .getOperatingSystemMXBean();
        double SystemCpuLoad = osmxb.getSystemCpuLoad();
        return SystemCpuLoad;
    }


    public static double getProcessCpuLoad() {
        OperatingSystemMXBean osmxb = (OperatingSystemMXBean) ManagementFactory
                .getOperatingSystemMXBean();
        double ProcessCpuLoad = osmxb.getProcessCpuLoad();
        return ProcessCpuLoad;
    }


    public static long getTotalMemorySize() {
        int kb = 1024;
        OperatingSystemMXBean osmxb = (OperatingSystemMXBean) ManagementFactory
                .getOperatingSystemMXBean();
        long totalMemorySize = osmxb.getTotalPhysicalMemorySize() / kb;
        return totalMemorySize;
    }


    public static long getFreePhysicalMemorySize() {
        int kb = 1024;
        OperatingSystemMXBean osmxb = (OperatingSystemMXBean) ManagementFactory
                .getOperatingSystemMXBean();
        long freePhysicalMemorySize = osmxb.getFreePhysicalMemorySize() / kb;
        return freePhysicalMemorySize;
    }


    public static long getUsedMemory() {
        int kb = 1024;
        OperatingSystemMXBean osmxb = (OperatingSystemMXBean) ManagementFactory
                .getOperatingSystemMXBean();
        long usedMemory = (osmxb.getTotalPhysicalMemorySize() - osmxb.getFreePhysicalMemorySize()) / kb;
        return usedMemory;
    }


    public static void test() {


        List<MemoryPoolMXBean> memoryPoolMXBeans = ManagementFactory.getMemoryPoolMXBeans();
        memoryPoolMXBeans.forEach((pool) -> {
            System.out.println(pool.getName());
            System.out.println(pool.getUsage());
        });


    }


    public static void main(String[] args) {

        System.err.println(getUsedMemory());

        System.err.println(getFreePhysicalMemorySize());

        System.err.println(getProcessCpuLoad());

        System.err.println(getSystemCpuLoad());

        test();

    }
}  