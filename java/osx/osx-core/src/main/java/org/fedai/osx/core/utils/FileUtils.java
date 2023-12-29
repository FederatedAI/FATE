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

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.channels.FileLock;

public class FileUtils {

    private static final Logger logger = LoggerFactory.getLogger(FileUtils.class);

    public static boolean writeFile(String context, File target) {
        BufferedWriter out = null;
        try {
            if (!target.exists()) {
                target.createNewFile();
            }
            out = new BufferedWriter(new FileWriter(target));
            out.write(context);
        } catch (IOException e) {
            logger.error(e.getMessage());
            return false;
        } finally {
            try {
                if (out != null) {
                    out.flush();
                    out.close();
                }
            } catch (IOException ex) {
                logger.error("write file error", ex);
            }
        }
        return true;
    }

    /**
     * Write string to file,
     * synchronize operation, exclusive lock
     */
    public static boolean writeStr2ReplaceFileSync(String str, String pathFile) throws Exception {
        File file = new File(pathFile);
        try {
            if (!file.exists()) {
                file.createNewFile();
            }
        } catch (IOException e) {
            logger.error("Failed to create the file. Check whether the path is valid and the read/write permission is correct");
            throw new IOException("Failed to create the file. Check whether the path is valid and the read/write permission is correct");
        }
        FileOutputStream fileOutputStream = null;
        FileChannel fileChannel = null;
        FileLock fileLock;
        try {

            /*
             * write file
             */
            fileOutputStream = new FileOutputStream(file);
            fileChannel = fileOutputStream.getChannel();

            try {
                fileLock = fileChannel.tryLock();// exclusive lock
            } catch (Exception e) {
                throw new IOException("another thread is writing ,refresh and try again");
            }
            if (fileLock != null) {
                fileChannel.write(ByteBuffer.wrap(str.getBytes()));
                if (fileLock.isValid()) {
                    fileLock.release(); // release-write-lock
                }
                if (file.length() != str.getBytes().length) {
                    throw new IOException("write successfully but the content was lost, reedit and try again");
                }
            }

        } catch (IOException e) {
            logger.error(e.getMessage());
            throw new IOException(e.getMessage());
        } finally {
            close(fileChannel);
            close(fileOutputStream);
        }
        return true;
    }

    public static void close(Closeable closeable) {
        if (closeable != null) {
            try {
                closeable.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    public static boolean createNewFile(String filePath) {
        return createNewFile(new File(filePath));
    }

    public static boolean createNewFile(File file) {
        try {
            if (!file.exists()) {
                if (!file.getParentFile().exists()) {
                    if (!file.getParentFile().mkdirs()) {
                        return false;
                    }
                }
                if (!file.createNewFile()) {
                    return false;
                }
            }
        } catch (IOException e) {
            logger.error("create file failed , path = {}", file.getAbsoluteFile());
            return false;
        }
        return true;
    }

}
