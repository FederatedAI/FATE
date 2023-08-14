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

package org.fedai.osx.core.jvm;

import java.lang.management.GarbageCollectorMXBean;
import java.lang.management.ManagementFactory;
import java.util.ArrayList;
import java.util.List;

public class JVMGCUtils {
    static private GarbageCollectorMXBean youngGC;
    static private GarbageCollectorMXBean fullGC;

    static {
        List<GarbageCollectorMXBean> gcMXBeanList = ManagementFactory.getGarbageCollectorMXBeans();
        for (final GarbageCollectorMXBean gcMXBean : gcMXBeanList) {
            String gcName = gcMXBean.getName();
            if (gcName == null) {
                continue;
            }
            //G1 Old Generation
            //Garbage collection optimized for short pausetimes Old Collector
            //Garbage collection optimized for throughput Old Collector
            //Garbage collection optimized for deterministic pausetimes Old Collector
            //G1 Young Generation
            //Garbage collection optimized for short pausetimes Young Collector
            //Garbage collection optimized for throughput Young Collector
            //Garbage collection optimized for deterministic pausetimes Young Collector
            if (fullGC == null &&
                    (gcName.endsWith("Old Collector")
                            || "ConcurrentMarkSweep".equals(gcName)
                            || "MarkSweepCompact".equals(gcName)
                            || "PS MarkSweep".equals(gcName))
            ) {
                fullGC = gcMXBean;
            } else if (youngGC == null &&
                    (gcName.endsWith("Young Generation")
                            || "ParNew".equals(gcName)
                            || "Copy".equals(gcName)
                            || "PS Scavenge".equals(gcName))
            ) {
                youngGC = gcMXBean;
            }
        }
    }//static

    //YGC名称
    static public String getYoungGCName() {
        return youngGC == null ? "" : youngGC.getName();
    }

    //YGC总次数
    static public long getYoungGCCollectionCount() {
        return youngGC == null ? 0 : youngGC.getCollectionCount();
    }

    //YGC总时间
    static public long getYoungGCCollectionTime() {
        return youngGC == null ? 0 : youngGC.getCollectionTime();
    }

    //FGC名称
    static public String getFullGCName() {
        return fullGC == null ? "" : fullGC.getName();
    }

    //FGC总次数
    static public long getFullGCCollectionCount() {
        return fullGC == null ? 0 : fullGC.getCollectionCount();
    }

    //FGC总次数
    static public long getFullGCCollectionTime() {
        return fullGC == null ? 0 : fullGC.getCollectionTime();
    }

    public static void main(String[] args) {
        List<List<Long>> listRoot = new ArrayList<List<Long>>();
        for (; ; ) {
//            System.out.println("=======================================================================");
//            System.out.println("getYoungGCName: " + JVMGCUtils.getYoungGCName());
//            System.out.println("getYoungGCCollectionCount: " + JVMGCUtils.getYoungGCCollectionCount());
//            System.out.println("getYoungGCCollectionTime: " + JVMGCUtils.getYoungGCCollectionTime());
//            System.out.println("getFullGCName: " + JVMGCUtils.getFullGCName());
//            System.out.println("getFullGCCollectionCount: " + JVMGCUtils.getFullGCCollectionCount());
//            System.out.println("getFullGCCollectionTime: " + JVMGCUtils.getFullGCCollectionTime());
            List<Long> list = new ArrayList<Long>(1000);
            listRoot.add(list);
            try {
                Thread.sleep(3000);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
            if (list.size() > 1) {
                list.remove(0);
            }
            Runtime.getRuntime().gc();
        }
    }
}
