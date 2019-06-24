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


import java.io.File;
import java.io.IOException;
import java.nio.file.*;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.Arrays;
import java.util.Random;

public class TestUtils {

    public static File tempDirectory() {
        return tempDirectory(null);
    }


    public static File tempDirectory(final String prefix) {
        return tempDirectory(null, prefix);
    }


    public static File tempDirectory(final Path parent, String prefix) {
        final File file;
        prefix = prefix == null ? "fdn-" : prefix;
        try {
            file = parent == null ?
                    Files.createTempDirectory(prefix).toFile() : Files.createTempDirectory(parent, prefix).toFile();
        } catch (final IOException ex) {
            throw new RuntimeException("Failed to create a temp dir", ex);
        }
        file.deleteOnExit();

        Runtime.getRuntime().addShutdownHook(new Thread() {
            @Override
            public void run() {
                try {
                    delete(file);
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        });

        return file;
    }

    public static void delete(final File file) throws IOException {
        if (file == null)
            return;
        Files.walkFileTree(file.toPath(), new SimpleFileVisitor<Path>() {
            @Override
            public FileVisitResult visitFileFailed(Path path, IOException exc) throws IOException {
                if (exc instanceof NoSuchFileException && path.toFile().equals(file))
                    return FileVisitResult.TERMINATE;
                throw exc;
            }

            @Override
            public FileVisitResult visitFile(Path path, BasicFileAttributes attrs) throws IOException {
                Files.delete(path);
                return FileVisitResult.CONTINUE;
            }

            @Override
            public FileVisitResult postVisitDirectory(Path path, IOException exc) throws IOException {
                Files.delete(path);
                return FileVisitResult.CONTINUE;
            }
        });
    }

    public static class RandomGenerator {
        final Random rnd;
        private final byte[] data;
        private int position;


        public RandomGenerator() {
            rnd = new Random(301);
            data = new byte[8 * 1024 * 1024 + rnd.nextInt(100)];
            rnd.nextBytes(data);
        }

        public byte[] generate(int length) {
            if (position + length > data.length) {
                position = 0;
                assert (length < data.length);
            }

            return Arrays.copyOfRange(data, position, position + length);
        }
    }


}
