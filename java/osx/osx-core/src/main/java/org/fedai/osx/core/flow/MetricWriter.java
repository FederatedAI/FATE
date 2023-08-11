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

package org.fedai.osx.core.flow;


import org.fedai.osx.core.config.MetaInfo;
import org.fedai.osx.core.utils.GetSystemInfo;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedOutputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.*;

public class MetricWriter {

    public static final String METRIC_BASE_DIR = MetaInfo.PROPERTY_USER_HOME + "/.fate/metric/";
    public static final String METRIC_FILE = "metrics.log";
    public static final String METRIC_FILE_INDEX_SUFFIX = ".idx";
    public static final Comparator<String> METRIC_FILE_NAME_CMP = new MetricFileNameComparator();
    private static final String CHARSET = "UTF-8";
    private final static int pid = GetSystemInfo.getPid();
    private final static boolean usePid = FlowCounterManager.USE_PID;
    private final DateFormat df = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
    Logger logger = LoggerFactory.getLogger(MetricWriter.class);
    String appName;
    private long timeSecondBase;
    private String baseDir;
    private String baseFileName;
    private File curMetricFile;
    private File curMetricIndexFile;
    private FileOutputStream outMetric;
    private DataOutputStream outIndex;
    private BufferedOutputStream outMetricBuf;
    private long singleFileSize;
    private int totalFileCount;
    private boolean append = false;
    /**
     * 秒级统计，忽略毫秒数。
     */
    private long lastSecond = -1;

    public MetricWriter(String appName, long singleFileSize) {
        this(appName, singleFileSize, 6);
    }

    public MetricWriter(String appName, long singleFileSize, int totalFileCount) {
        this.appName = appName;
        if (singleFileSize <= 0 || totalFileCount <= 0) {
            throw new IllegalArgumentException();
        }

        this.baseDir = METRIC_BASE_DIR;
        File dir = new File(baseDir);
        if (!dir.exists()) {
            dir.mkdirs();
        }

        long time = System.currentTimeMillis();
        this.lastSecond = time / 1000;
        this.singleFileSize = singleFileSize;
        this.totalFileCount = totalFileCount;
        try {
            this.timeSecondBase = df.parse("1970-01-01 00:00:00").getTime() / 1000;
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * Get all metric files' name in {@code baseDir}. The file name must like
     * <pre>
     * baseFileName + ".yyyy-MM-dd.number"
     * </pre>
     * and not endsWith {@link #METRIC_FILE_INDEX_SUFFIX} or ".lck".
     *
     * @param baseDir      the directory to search.
     * @param baseFileName the file name pattern.
     * @return the metric files' absolute path({@link File#getAbsolutePath()})
     * @throws Exception
     */
    static List<String> listMetricFiles(String baseDir, String baseFileName) throws Exception {
        List<String> list = new ArrayList<String>();
        File baseFile = new File(baseDir);
        File[] files = baseFile.listFiles();
        if (files == null) {
            return list;
        }
        for (File file : files) {
            String fileName = file.getName();
            if (file.isFile()
                    && fileNameMatches(fileName, baseFileName)
                    && !fileName.endsWith(METRIC_FILE_INDEX_SUFFIX)
                    && !fileName.endsWith(".lck")) {
                list.add(file.getAbsolutePath());
            }
        }
        Collections.sort(list, METRIC_FILE_NAME_CMP);
        return list;
    }

    /**
     * Test whether fileName matches baseFileName. fileName matches baseFileName when
     * <pre>
     * fileName = baseFileName + ".yyyy-MM-dd.number"
     * </pre>
     *
     * @param fileName     file name
     * @param baseFileName base file name.
     * @return if fileName matches baseFileName return true, else return false.
     */
    public static boolean fileNameMatches(String fileName, String baseFileName) {
        String matchFileName = baseFileName;
        String part = "";
        if (usePid) {
            matchFileName = matchFileName.substring(0, matchFileName.indexOf(String.valueOf(pid)));
            //  System.err.println(matchFileName);
            // fileName: serving-metrics.log.pid71860.2020-06-12
            // baseFileName: serving-metrics.log.pid71860
            if (fileName.startsWith(matchFileName) && !fileName.startsWith(matchFileName + ".")) {
                part = fileName.substring(matchFileName.length()); // 71860.2020-06-12
                String[] split = part.split("\\.");
                if (!split[0].equals(String.valueOf(pid))) {
                    return false;
                }
                part = part.substring(part.indexOf(".")); // .2020-06-12
            }

        } else {
            if (fileName.startsWith(matchFileName)) {
                part = fileName.substring(matchFileName.length());
            }
        }
        // part is like: ".yyyy-MM-dd.number", eg. ".2018-12-24.11"
        return part.matches("\\.[0-9]{4}-[0-9]{2}-[0-9]{2}(\\.[0-9]*)?");
    }

    public static boolean fileNameAllMatches(String fileName, String baseFileName) {
        String matchFileName = baseFileName;
        String part = "";
        if (usePid) {
            matchFileName = matchFileName.substring(0, matchFileName.indexOf(String.valueOf(pid)));
            // System.err.println(matchFileName);
            if (fileName.startsWith(matchFileName) && !fileName.startsWith(matchFileName + ".")) {

                part = fileName.substring(matchFileName.length());
                part = part.substring(part.indexOf("."));
            }

        } else {
            if (fileName.startsWith(matchFileName)) {
                part = fileName.substring(matchFileName.length());
            }
        }
        // part is like: ".yyyy-MM-dd.number", eg. ".2018-12-24.11"

        return part.matches("\\.[0-9]{4}-[0-9]{2}-[0-9]{2}");
    }

    /**
     * Form metric file name use the specific appName and pid. Note that only
     * form the file name, not include path.
     * <p>
     * Note: {@link MetricFileNameComparator}'s implementation relays on the metric file name,
     * we should be careful when changing the metric file name.
     *
     * @param appName
     * @param pid
     * @return metric file name.
     */
    public static String formMetricFileName(String appName, int pid) {
        if (appName == null) {
            appName = "";
        }
        // dot is special char that should be replaced.
        final String dot = ".";
        final String separator = "-";
        if (appName.contains(dot)) {
            appName = appName.replace(dot, separator);
        }
        String name = appName + separator + METRIC_FILE;
        if (usePid) {
            name += ".pid" + pid;
        }
        return name;
    }

    /**
     * Form index file name of the {@code metricFileName}
     *
     * @param metricFileName
     * @return the index file name of the metricFileName
     */
    public static String formIndexFileName(String metricFileName) {
        return metricFileName + METRIC_FILE_INDEX_SUFFIX;
    }

    /**
     * 如果传入了time，就认为nodes中所有的时间时间戳都是time.
     *
     * @param time
     * @param nodes
     */
    public synchronized void write(long time, List<MetricNode> nodes) throws Exception {
        if (nodes == null) {
            return;
        }
        for (MetricNode node : nodes) {
            node.setTimestamp(time);
        }


        // first write, should create file
        if (curMetricFile == null) {
            baseFileName = formMetricFileName(appName, pid);
            closeAndNewFile(nextFileNameOfDay(time));
        }
        if (!(curMetricFile.exists() && curMetricIndexFile.exists())) {
            closeAndNewFile(nextFileNameOfDay(time));
        }

        long second = time / 1000;
        if (second < lastSecond) {
            // 时间靠前的直接忽略，不应该发生。
        } else if (second == lastSecond) {
            for (MetricNode node : nodes) {
                outMetricBuf.write(node.toFatString().getBytes(CHARSET));
            }
            outMetricBuf.flush();
            if (!validSize()) {
                closeAndNewFile(nextFileNameOfDay(time));
            }
        } else {
            writeIndex(second, outMetric.getChannel().position());
            if (isNewDay(lastSecond, second)) {
                closeAndNewFile(nextFileNameOfDay(time));
                for (MetricNode node : nodes) {
                    outMetricBuf.write(node.toFatString().getBytes(CHARSET));
                }
                outMetricBuf.flush();
                if (!validSize()) {
                    closeAndNewFile(nextFileNameOfDay(time));
                }
            } else {
                for (MetricNode node : nodes) {
                    outMetricBuf.write(node.toFatString().getBytes(CHARSET));
                }
                outMetricBuf.flush();
                if (!validSize()) {
                    closeAndNewFile(nextFileNameOfDay(time));
                }
            }
            lastSecond = second;
        }
    }

    public synchronized void close() throws Exception {
        if (outMetricBuf != null) {
            outMetricBuf.close();
        }
        if (outIndex != null) {
            outIndex.close();
        }
    }

    private void writeIndex(long time, long offset) throws Exception {
        outIndex.writeLong(time);
        outIndex.writeLong(offset);
        outIndex.flush();
    }

    private String nextFileNameOfDay(long time) {
        List<String> list = new ArrayList<String>();
        File baseFile = new File(baseDir);
        DateFormat fileNameDf = new SimpleDateFormat("yyyy-MM-dd");
        String dateStr = fileNameDf.format(new Date(time));
        String fileNameModel = baseFileName + "." + dateStr;
        for (File file : baseFile.listFiles()) {
            String fileName = file.getName();
            if (fileName.contains(fileNameModel)
                    && !fileName.endsWith(METRIC_FILE_INDEX_SUFFIX)
                    && !fileName.endsWith(".lck")) {
                list.add(file.getAbsolutePath());
            }
        }
        Collections.sort(list, METRIC_FILE_NAME_CMP);
        if (list.isEmpty()) {
            return baseDir + fileNameModel;
        }
        String last = list.get(list.size() - 1);
        int n = 0;
        String[] strs = last.split("\\.");
        if (strs.length > 0 && strs[strs.length - 1].matches("[0-9]{1,10}")) {
            n = Integer.parseInt(strs[strs.length - 1]);
        }
        return baseDir + fileNameModel + "." + (n + 1);
    }

    public void removeMoreFiles() throws Exception {
        List<String> list = listMetricFiles(baseDir, baseFileName);
        if (list == null || list.isEmpty()) {
            return;
        }
        for (int i = 0; i < list.size() - totalFileCount + 1; i++) {
            String fileName = list.get(i);
            String indexFile = formIndexFileName(fileName);
            new File(fileName).delete();
            logger.info("removing metric file: " + fileName);
            new File(indexFile).delete();
//            RecordLog.info("[MetricWriter] Removing metric index file: " + indexFile);
        }
    }

    public void removeAllFiles() throws Exception {
        List<String> list = listMetricFiles(baseDir, baseFileName);
        if (list == null || list.isEmpty()) {
            return;
        }
        for (int i = 0; i < list.size(); i++) {
            String fileName = list.get(i);
            if (fileName.indexOf("pid" + pid + ".") > 0) {
                String indexFile = formIndexFileName(fileName);
                try {
                    new File(fileName).delete();
                } catch (Exception e) {

                }
                System.err.println("removing metric file: " + fileName);
                try {
                    new File(indexFile).delete();
                } catch (Exception e) {

                }
                System.err.println("removing metric file: " + indexFile);
            } else {
                System.err.println("metric file " + fileName + " is not match");
            }

        }
    }

    private void closeAndNewFile(String fileName) throws Exception {
        removeMoreFiles();
        if (outMetricBuf != null) {
            outMetricBuf.close();
        }
        if (outIndex != null) {
            outIndex.close();
        }
        outMetric = new FileOutputStream(fileName, append);
        outMetricBuf = new BufferedOutputStream(outMetric);
        curMetricFile = new File(fileName);
        String idxFile = formIndexFileName(fileName);
        curMetricIndexFile = new File(idxFile);
        outIndex = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(idxFile, append)));
        //RecordLog.info("[MetricWriter] New metric file created: " + fileName);
        //RecordLog.info("[MetricWriter] New metric index file created: " + idxFile);
    }

    private boolean validSize() throws Exception {
        long size = outMetric.getChannel().size();
        return size < singleFileSize;
    }

    private boolean isNewDay(long lastSecond, long second) {
        long lastDay = (lastSecond - timeSecondBase) / 86400;
        long newDay = (second - timeSecondBase) / 86400;
        return newDay > lastDay;
    }

    /**
     * A comparator for metric file name. Metric file name is like: <br/>
     * <pre>
     * metrics.log.2018-03-06
     * metrics.log.2018-03-07
     * metrics.log.2018-03-07.10
     * metrics.log.2018-03-06.100
     * </pre>
     * <p>
     * File name with the early date is smaller, if date is same, the one with the small file number is smaller.
     * Note that if the name is an absolute path, only the fileName({@link File#getName()}) part will be considered.
     * So the above file names should be sorted as: <br/>
     * <pre>
     * metrics.log.2018-03-06
     * metrics.log.2018-03-06.100
     * metrics.log.2018-03-07
     * metrics.log.2018-03-07.10
     *
     * </pre>
     * </p>
     */
    private static final class MetricFileNameComparator implements Comparator<String> {
        private final String pid = "pid";

        @Override
        public int compare(String o1, String o2) {
            String name1 = new File(o1).getName();
            String name2 = new File(o2).getName();
            String dateStr1 = name1.split("\\.")[2];
            String dateStr2 = name2.split("\\.")[2];
            // in case of file name contains pid, skip it, like Sentinel-Admin-metrics.log.pid22568.2018-12-24
            if (dateStr1.startsWith(pid)) {
                dateStr1 = name1.split("\\.")[3];
                dateStr2 = name2.split("\\.")[3];
            }

            // compare date first
            int t = dateStr1.compareTo(dateStr2);
            if (t != 0) {
                return t;
            }

            // same date, compare file number
            t = name1.length() - name2.length();
            if (t != 0) {
                return t;
            }
            return name1.compareTo(name2);
        }
    }
}
