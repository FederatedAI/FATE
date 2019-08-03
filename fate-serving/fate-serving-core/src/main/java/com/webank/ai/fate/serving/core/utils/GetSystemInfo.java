package com.webank.ai.fate.serving.core.utils;



import com.sun.management.OperatingSystemMXBean;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;


import java.lang.management.ManagementFactory;
import java.lang.management.MemoryPoolMXBean;
import java.net.*;
import java.util.Enumeration;
import java.util.List;

public class GetSystemInfo {

    private static final Logger LOGGER = LogManager.getLogger();


    public static  String  localIp;

    static  {
        localIp = getLocalIp();
    }

    public static String getLocalIp() {

        String sysType = System.getProperties().getProperty("os.name");
        String ip;

        try {
        if (sysType.toLowerCase().startsWith("win")) {
            String localIP = null;

                localIP = InetAddress.getLocalHost().getHostAddress();

            if (localIP != null) {
                return localIP;
            }
        } else {
            ip = getIpByEthNum("eth0");
            if (ip != null) {
                return ip;


            }
        }
        } catch (Throwable  e) {
            LOGGER.error(e.getMessage(), e);
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
            LOGGER.error(e.getMessage(), e);
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






}  

