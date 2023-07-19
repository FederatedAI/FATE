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

package com.osx.core.jvm;

import java.lang.management.ManagementFactory;
import java.lang.management.MemoryMXBean;
import java.lang.management.MemoryPoolMXBean;
import java.lang.management.MemoryUsage;
import java.util.List;

/**
 * 类描述：JVM内存信息工具类
 **/
public class JVMMemoryUtils {
    static private MemoryMXBean memoryMXBean;
    static private MemoryPoolMXBean edenSpaceMxBean;
    static private MemoryPoolMXBean survivorSpaceMxBean;
    static private MemoryPoolMXBean oldGenMxBean;
    static private MemoryPoolMXBean permGenMxBean;
    static private MemoryPoolMXBean codeCacheMxBean;

    static {
        memoryMXBean = ManagementFactory.getMemoryMXBean();

        List<MemoryPoolMXBean> memoryPoolMXBeanList = ManagementFactory.getMemoryPoolMXBeans();
        for (final MemoryPoolMXBean memoryPoolMXBean : memoryPoolMXBeanList) {
            String poolName = memoryPoolMXBean.getName();
            if (poolName == null) {
                continue;
            }
            // 官方JVM(HotSpot)提供的MemoryPoolMXBean
            // JDK1.7/1.8 Eden区内存池名称： "Eden Space" 或  "PS Eden Space"、 “G1 Eden Space”(和垃圾收集器有关)
            // JDK1.7/1.8 Survivor区内存池名称："Survivor Space" 或 "PS Survivor Space"、“G1 Survivor Space”(和垃圾收集器有关)
            // JDK1.7  老区内存池名称： "Tenured Gen"
            // JDK1.8  老区内存池名称："Old Gen" 或 "PS Old Gen"、“G1 Old Gen”(和垃圾收集器有关)
            // JDK1.7  方法/永久区内存池名称： "Perm Gen" 或 "PS Perm Gen"(和垃圾收集器有关)
            // JDK1.8  方法/永久区内存池名称："Metaspace"(注意：不在堆内存中)
            // JDK1.7/1.8  CodeCache区内存池名称： "Code Cache"
            if (edenSpaceMxBean == null && poolName.endsWith("Eden Space")) {
                edenSpaceMxBean = memoryPoolMXBean;
            } else if (survivorSpaceMxBean == null && poolName.endsWith("Survivor Space")) {
                survivorSpaceMxBean = memoryPoolMXBean;
            } else if (oldGenMxBean == null && (poolName.endsWith("Tenured Gen") || poolName.endsWith("Old Gen"))) {
                oldGenMxBean = memoryPoolMXBean;
            } else if (permGenMxBean == null && (poolName.endsWith("Perm Gen") || poolName.endsWith("Metaspace"))) {
                permGenMxBean = memoryPoolMXBean;
            } else if (codeCacheMxBean == null && poolName.endsWith("Code Cache")) {
                codeCacheMxBean = memoryPoolMXBean;
            }
        }
    }// static

    /**
     * 获取堆内存情况
     *
     * @return 不能获取到返回null
     */
    static public JVMMemoryUsage getHeapMemoryUsage() {
        if (memoryMXBean != null) {
            final MemoryUsage usage = memoryMXBean.getHeapMemoryUsage();
            if (usage != null) {
                return new JVMMemoryUsage(usage);
            }
        }
        return null;
    }

    /**
     * 获取堆外内存情况
     *
     * @return 不能获取到返回null
     */
    static public JVMMemoryUsage getNonHeapMemoryUsage() {
        if (memoryMXBean != null) {
            final MemoryUsage usage = memoryMXBean.getNonHeapMemoryUsage();
            if (usage != null) {
                return new JVMMemoryUsage(usage);
            }
        }
        return null;
    }

    /**
     * 获取Eden区内存情况
     *
     * @return 不能获取到返回null
     */
    static public JVMMemoryUsage getEdenSpaceMemoryUsage() {
        return getMemoryPoolUsage(edenSpaceMxBean);
    }

    /**
     * 获取Eden区内存峰值（从启动或上一次重置开始统计），并重置
     *
     * @return 不能获取到返回null
     */
    static public JVMMemoryUsage getAndResetEdenSpaceMemoryPeakUsage() {
        return getAndResetMemoryPoolPeakUsage(edenSpaceMxBean);
    }

    /**
     * 获取Survivor区内存情况
     *
     * @return 不能获取到返回null
     */
    static public JVMMemoryUsage getSurvivorSpaceMemoryUsage() {
        return getMemoryPoolUsage(survivorSpaceMxBean);
    }

    /**
     * 获取Survivor区内存峰值（从启动或上一次重置开始统计），并重置
     *
     * @return 不能获取到返回null
     */
    static public JVMMemoryUsage getAndResetSurvivorSpaceMemoryPeakUsage() {
        return getAndResetMemoryPoolPeakUsage(survivorSpaceMxBean);
    }

    /**
     * 获取老区内存情况
     *
     * @return 不能获取到返回null
     */
    static public JVMMemoryUsage getOldGenMemoryUsage() {
        return getMemoryPoolUsage(oldGenMxBean);
    }

    /**
     * 获取老区内存峰值（从启动或上一次重置开始统计），并重置
     *
     * @return 不能获取到返回null
     */
    static public JVMMemoryUsage getAndResetOldGenMemoryPeakUsage() {
        return getAndResetMemoryPoolPeakUsage(oldGenMxBean);
    }

    /**
     * 获取永久区/方法区内存情况
     *
     * @return 不能获取到返回null
     */
    static public JVMMemoryUsage getPermGenMemoryUsage() {
        return getMemoryPoolUsage(permGenMxBean);
    }

    /**
     * 获取永久区/方法区内存峰值（从启动或上一次重置开始统计），并重置
     *
     * @return 不能获取到返回null
     */
    static public JVMMemoryUsage getAndResetPermGenMemoryPeakUsage() {
        return getAndResetMemoryPoolPeakUsage(permGenMxBean);
    }

    /**
     * 获取CodeCache区内存情况
     *
     * @return 不能获取到返回null
     */
    static public JVMMemoryUsage getCodeCacheMemoryUsage() {
        return getMemoryPoolUsage(codeCacheMxBean);
    }

    /**
     * 获取CodeCache区内存峰值（从启动或上一次重置开始统计），并重置
     *
     * @return 不能获取到返回null
     */
    static public JVMMemoryUsage getAndResetCodeCacheMemoryPeakUsage() {
        return getAndResetMemoryPoolPeakUsage(codeCacheMxBean);
    }

    static private JVMMemoryUsage getMemoryPoolUsage(MemoryPoolMXBean memoryPoolMXBean) {
        if (memoryPoolMXBean != null) {
            final MemoryUsage usage = memoryPoolMXBean.getUsage();
            if (usage != null) {
                return new JVMMemoryUsage(usage);
            }
        }
        return null;
    }

    static public String getProcessId() {
        String processName = ManagementFactory.getRuntimeMXBean().getName();
        String processId = processName.substring(0, processName.indexOf('@'));
        return processId;
    }

    static private JVMMemoryUsage getAndResetMemoryPoolPeakUsage(MemoryPoolMXBean memoryPoolMXBean) {
        if (memoryPoolMXBean != null) {
            final MemoryUsage usage = memoryPoolMXBean.getPeakUsage();
            if (usage != null) {
                memoryPoolMXBean.resetPeakUsage();
                return new JVMMemoryUsage(usage);
            }
        }
        return null;
    }

    public static void main(String[] args) {

        System.err.println(JVMMemoryUtils.getProcessId());
        // 	List<List<Long>> listRoot = new ArrayList<List<Long>>();
        // 	for(;;) {
//    		System.out.println("=======================================================================");
//	        System.out.println("getHeapMemoryUsage: " + JVMMemoryUtils.getHeapMemoryUsage());
//	        System.out.println("getNonHeapMemoryUsage: " + JVMMemoryUtils.getNonHeapMemoryUsage());
//	        System.out.println("getEdenSpaceMemoryUsage: " + JVMMemoryUtils.getEdenSpaceMemoryUsage());
//	        System.out.println("getAndResetEdenSpaceMemoryPeakUsage: " + JVMMemoryUtils.getAndResetEdenSpaceMemoryPeakUsage());
//	        System.out.println("getSurvivorSpaceMemoryUsage: " + JVMMemoryUtils.getSurvivorSpaceMemoryUsage());
//	        System.out.println("getAndResetSurvivorSpaceMemoryPeakUsage: " + JVMMemoryUtils.getAndResetSurvivorSpaceMemoryPeakUsage());
//	        System.out.println("getOldGenMemoryUsage: " + JVMMemoryUtils.getOldGenMemoryUsage());
//	        System.out.println("getAndResetOldGenMemoryPeakUsage: " + JVMMemoryUtils.getAndResetOldGenMemoryPeakUsage());
//	        System.out.println("getPermGenMemoryUsage: " + JVMMemoryUtils.getPermGenMemoryUsage());
//	        System.out.println("getAndResetPermGenMemoryPeakUsage: " + JVMMemoryUtils.getAndResetPermGenMemoryPeakUsage());
//	        System.out.println("getCodeCacheMemoryUsage: " + JVMMemoryUtils.getCodeCacheMemoryUsage());
//	        System.out.println("getAndResetCodeCacheMemoryPeakUsage: " + JVMMemoryUtils.getAndResetCodeCacheMemoryPeakUsage());
//	        List<Long> list = new ArrayList<Long>(10000);
//	        listRoot.add(list);
//	        try {
//				Thread.sleep(3000);
//			} catch (InterruptedException e) {
//				e.printStackTrace();
//			}
//
//	        if(list.size() > 1) {
//	        	list.remove(0);
//	        }
//	        Runtime.getRuntime().gc();
//    	}
    }

    /**
     * JVM内存区域使用情况。</br>
     * <pre>
     * init：初始内存大小（字节）
     * used：当前使用内存大小（字节）
     * committed：已经申请分配的内存大小（字节）
     * max：最大内存大小（字节）
     * usedPercent：已经申请分配内存与最大内存大小的百分比
     * </pre>
     *
     * @author tangjiyu
     */
    static public class JVMMemoryUsage {
        //初始内存大小（字节）
        private long init;
        //当前使用内存大小（字节）
        private long used;
        //已经申请分配的内存大小（字节）
        private long committed;
        //最大内存大小（字节）
        private long max;
        //已经申请分配内存与最大内存大小的百分比
        private float usedPercent;

        public JVMMemoryUsage() {
        }

        public JVMMemoryUsage(MemoryUsage memoryUsage) {
            this.setMemoryUsage(memoryUsage);
            //this(memoryUsage.getInit(), memoryUsage.getUsed(), memoryUsage.getCommitted(), memoryUsage.getMax());
        }

        public JVMMemoryUsage(long init, long used, long committed, long max) {
            super();
            this.setMemoryUsage(init, used, committed, max);
        }

        private void setMemoryUsage(MemoryUsage memoryUsage) {
            if (memoryUsage != null) {
                this.setMemoryUsage(memoryUsage.getInit(), memoryUsage.getUsed(), memoryUsage.getCommitted(), memoryUsage.getMax());
            } else {
                this.setMemoryUsage(0, 0, 0, 0);
            }
        }

        private void setMemoryUsage(long init, long used, long committed, long max) {
            this.init = init;
            this.used = used;
            this.committed = committed;
            this.max = max;
            if (this.used > 0 && max > 0) {
                this.usedPercent = used * Float.valueOf("1.0") / max;
            } else {
                this.usedPercent = 0;
            }
        }

        public long getInit() {
            return init;
        }

        public long getUsed() {
            return used;
        }

        public long getCommitted() {
            return committed;
        }

        public long getMax() {
            return max;
        }

        public float getUsedPercent() {
            return usedPercent;
        }

        @Override
        public String toString() {
            StringBuffer buf = new StringBuffer();
            buf.append("init = " + init + "(" + (init >> 10) + "K) ");
            buf.append("used = " + used + "(" + (used >> 10) + "K) ");
            buf.append("committed = " + committed + "(" +
                    (committed >> 10) + "K) ");
            buf.append("max = " + max + "(" + (max >> 10) + "K)");
            buf.append("usedPercent = " + usedPercent);
            return buf.toString();
        }
    }
}
