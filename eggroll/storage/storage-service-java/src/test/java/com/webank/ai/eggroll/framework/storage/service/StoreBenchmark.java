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

package com.webank.ai.eggroll.framework.storage.service;

import com.google.common.base.Splitter;
import com.google.common.collect.ImmutableList;
import com.google.common.io.CharStreams;
import com.google.common.io.Files;
import com.webank.ai.eggroll.core.io.KeyValue;
import com.webank.ai.eggroll.core.io.KeyValueIterator;
import com.webank.ai.eggroll.core.io.KeyValueStore;
import com.webank.ai.eggroll.core.model.Bytes;

import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.concurrent.TimeUnit;

import static com.webank.ai.eggroll.framework.storage.service.StoreBenchmark.Order.RANDOM;
import static com.webank.ai.eggroll.framework.storage.service.StoreBenchmark.Order.SEQUENTIAL;
import static java.nio.charset.StandardCharsets.UTF_8;

public class StoreBenchmark {
    private final List<String> benchmarks;
    private final int num;
    private final int valueSize;
    TestUtils.RandomGenerator generator;
    private KeyValueStore<Bytes, byte[]> store;
    private long startTime;
    private long bytes;
    private String message;
    private int reads;
    private String postMessage;
    private double lastOpFinish;
    private int done;
    private int nextReport;
    private Random random;

    public StoreBenchmark(Map<Flag, Object> flags, KeyValueStore kvStore) {
        benchmarks = (List<String>) flags.get(Flag.benchmarks);
        num = (Integer) flags.get(Flag.num);
        reads = (Integer) (flags.get(Flag.reads) == null ? flags.get(Flag.num) : flags.get(Flag.reads));
        valueSize = (Integer) flags.get(Flag.value_size);
        bytes = 0;
        this.store = kvStore;
        random = new Random(301);
        generator = new TestUtils.RandomGenerator();
    }

    public static byte[] formatNumber(long n) {
        byte[] slice = new byte[16];

        int i = 15;
        while (n > 0) {
            slice[i--] = (byte) ((long) '0' + (n % 10));
            n /= 10;
        }
        while (i >= 0) {
            slice[i--] = '0';
        }
        return slice;
    }

    public void run() throws IOException {
        printHeader();

        for (String benchmark : benchmarks) {
            start();

            boolean known = true;

            if (benchmark.equals("fillseq")) {
                write(SEQUENTIAL, num, valueSize, 1);
            } else if (benchmark.equals("fillbatch")) {
                write(SEQUENTIAL, num, valueSize, 1000);
            } else if (benchmark.equals("fillrandom")) {
                write(RANDOM, num, valueSize, 1);
            } else if (benchmark.equals("overwrite")) {
                write(RANDOM, num, valueSize, 1);
            } else if (benchmark.equals("fillsync")) {
                write(RANDOM, num / 1000, valueSize, 1);
            } else if (benchmark.equals("fill100K")) {
                write(RANDOM, num / 1000, 100 * 1000, 1);
            } else if (benchmark.equals("readseq")) {
                readSequential();
            } else if (benchmark.equals("readrandom")) {
                readRandom();
            } else if (benchmark.equals("readhot")) {
                readHot();
            } else if (benchmark.equals("readrandomsmall")) {
                int n = reads;
                reads /= 1000;
                readRandom();
                reads = n;
            } else {
                known = false;
                System.err.println("Unknown benchmark: " + benchmark);
            }
            if (known) {
                stop(benchmark);
            }
        }
    }

    private void readSequential() {
        for (int loops = 0; loops < 5; loops++) {
            KeyValueIterator<Bytes, byte[]> iterator = store.all();
            for (int i = 0; i < reads && iterator.hasNext(); i++) {
                KeyValue<Bytes, byte[]> entry = iterator.next();
                bytes += entry.key.get().length + entry.value.length;
                finishedSingleOp();
            }
            iterator.close();
        }
    }

    private void readRandom() {
        for (int i = 0; i < reads; i++) {
            byte[] key = formatNumber(random.nextInt(num));
            byte[] value = store.get(Bytes.wrap(key));
            if (value == null) {
                throw new NullPointerException(String.format("db.get(%s) is null", new String(key, UTF_8)));
            }
            bytes += key.length + value.length;
            finishedSingleOp();
        }
    }

    private void readHot() {
        int range = (num + 99) / 100;
        for (int i = 0; i < reads; i++) {
            byte[] key = formatNumber(random.nextInt(range));
            byte[] value = store.get(Bytes.wrap(key));
            bytes += key.length + value.length;
            finishedSingleOp();
        }
    }

    private void write(Order order, int numEntries, int valueSize, int entriesPerBatch)
            throws IOException {

        if (numEntries != num) {
            message = String.format("(%d ops)", numEntries);
        }

        for (int i = 0; i < numEntries; i += entriesPerBatch) {
            List<KeyValue<Bytes, byte[]>> batch = new ArrayList<>();
            for (int j = 0; j < entriesPerBatch; j++) {
                int k = (order == SEQUENTIAL) ? i + j : random.nextInt(num);
                Bytes key = Bytes.wrap(formatNumber(k));
                batch.add(new KeyValue<>(key, generator.generate(valueSize)));
                bytes += valueSize + key.get().length;
                finishedSingleOp();
            }
            store.putAll(batch);
        }
    }

    private void finishedSingleOp() {
        done++;
        if (done >= nextReport) {
            if (nextReport < 1000) {
                nextReport += 100;
            } else if (nextReport < 5000) {
                nextReport += 500;
            } else if (nextReport < 10000) {
                nextReport += 1000;
            } else if (nextReport < 50000) {
                nextReport += 5000;
            } else if (nextReport < 100000) {
                nextReport += 10000;
            } else if (nextReport < 500000) {
                nextReport += 50000;
            } else {
                nextReport += 100000;
            }
            System.out.printf("... finished %d ops%30s\r", done, "");

        }
    }

    private void printHeader()
            throws IOException {
        int kKeySize = 16;
        printEnvironment();
        System.out.printf("Keys:       %d bytes each\n", kKeySize);
        System.out.printf("Values:     %d bytes each\n", valueSize);
        System.out.printf("Entries:    %d\n", num);
        System.out.printf("RawSize:    %.1f MB (estimated)\n",
                ((kKeySize + valueSize) * num) / 1048576.0);
        System.out.printf("------------------------------------------------\n");
    }

    private void printEnvironment() throws IOException {
        System.out.printf("Store:      %s\n", store.getClass().getName());

        System.out.printf("Date:       %tc\n", new Date());

        File cpuInfo = new File("/proc/cpuinfo");
        if (cpuInfo.canRead()) {
            int numberOfCpus = 0;
            String cpuType = null;
            String cacheSize = null;
            for (String line : CharStreams.readLines(Files.newReader(cpuInfo, UTF_8))) {
                ImmutableList<String> parts = ImmutableList.copyOf(Splitter.on(':').omitEmptyStrings().trimResults().limit(2).split(line));
                if (parts.size() != 2) {
                    continue;
                }
                String key = parts.get(0);
                String value = parts.get(1);

                if (key.equals("model name")) {
                    numberOfCpus++;
                    cpuType = value;
                } else if (key.equals("cache size")) {
                    cacheSize = value;
                }
            }
            System.out.printf("CPU:        %d * %s\n", numberOfCpus, cpuType);
            System.out.printf("CPUCache:   %s\n", cacheSize);
        }
    }

    private void start() {
        startTime = System.nanoTime();
        bytes = 0;
        message = null;
        lastOpFinish = startTime;
        done = 0;
        nextReport = 100;
    }


    private void stop(String benchmark) {
        long endTime = System.nanoTime();
        double elapsedSeconds = 1.0d * (endTime - startTime) / TimeUnit.SECONDS.toNanos(1);

        if (done < 1) {
            done = 1;
        }

        if (bytes > 0) {
            String rate = String.format("%6.1f MB/s", (bytes / 1048576.0) / elapsedSeconds);
            if (message != null) {
                message = rate + " " + message;
            } else {
                message = rate;
            }
        } else if (message == null) {
            message = "";
        }

        System.out.printf("%-12s : %11.5f micros/op;%s%s\n",
                benchmark,
                elapsedSeconds * 1.0e6 / done,
                (message == null ? "" : " "),
                message);


        if (postMessage != null) {
            System.out.printf("\n%s\n", postMessage);
            postMessage = null;
        }

    }

    enum Order {
        SEQUENTIAL,
        RANDOM
    }

    public enum Flag {
        benchmarks(ImmutableList.of(
                "fillseq",
                "fillbatch",
                "fillsync",
                "fillrandom",
                "overwrite",
                "fillseq",
                "readrandom",
                "readrandom",
                "readseq",
                "readrandom",
                "readseq",
                "fill100K"
        )) {
            @Override
            public Object parseValue(String value) {
                return ImmutableList.copyOf(Splitter.on(",").trimResults().omitEmptyStrings().split(value));
            }
        },

        type("LevelDB") {
            @Override
            public Object parseValue(String value) {
                return value;
            }
        },

        num(1000000) {
            @Override
            public Object parseValue(String value) {
                return Integer.parseInt(value);
            }
        },

        reads(null) {
            @Override
            public Object parseValue(String value) {
                return Integer.parseInt(value);
            }
        },

        value_size(100) {
            @Override
            public Object parseValue(String value) {
                return Integer.parseInt(value);
            }
        };

        private final Object defaultValue;

        Flag(Object defaultValue) {
            this.defaultValue = defaultValue;
        }

        protected abstract Object parseValue(String value);

        public Object getDefaultValue() {
            return defaultValue;
        }
    }


}
