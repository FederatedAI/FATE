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

package com.webank.ai.fate.eggroll.storage.service.model;

import com.webank.ai.fate.core.error.exception.InvalidStateStoreException;
import com.webank.ai.fate.core.error.exception.ProcessorStateException;
import com.webank.ai.fate.core.io.KeyValue;
import com.webank.ai.fate.core.io.KeyValueIterator;
import com.webank.ai.fate.core.io.KeyValueStore;
import com.webank.ai.fate.core.io.StoreInfo;
import com.webank.ai.fate.core.model.Bytes;
import com.webank.ai.fate.core.utils.AbstractIterator;
import com.webank.ai.fate.core.utils.ErrorUtils;
import io.grpc.stub.StreamObserver;
import org.apache.commons.io.FileUtils;
import org.apache.commons.lang3.exception.ExceptionUtils;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.iq80.leveldb.DBException;
import org.lmdbjava.*;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;


public class LMDBStore implements KeyValueStore<Bytes, byte[]> {
    public static final String DATA_DIR = "data.dir";
    private static final String TOSTRING_FORMAT = "LMDBStore : %s";
    private static Logger LOGGER = LogManager.getLogger(LMDBStore.class);
    private final StoreInfo storeInfo;
    private final Set<KeyValueIterator> openIterators = Collections.synchronizedSet(new HashSet<KeyValueIterator>());
    private File dbDir;
    private volatile boolean open = false;
    private Env<byte[]> env;
    private Dbi<byte[]> dbi;

    private ErrorUtils errorUtils;

    public LMDBStore(StoreInfo info) {
        this.storeInfo = info;
        this.errorUtils = new ErrorUtils();
    }

    @Override
    public void put(Bytes key, byte[] value) {
        Objects.requireNonNull(key, "key cannot be null");
        if (value == null || value.length == 0) {
            dbi.delete(key.get());
        } else {
            dbi.put(key.get(), value);
        }
    }

    @Override
    public byte[] putIfAbsent(Bytes key, byte[] value) {
        Objects.requireNonNull(key, "key cannot be null");

        try (Txn<byte[]> txn = env.txnWrite()) {
            byte[] oldValue = dbi.get(txn, key.get());
            if (oldValue == null) {
                if (value == null || value.length == 0) {
                    dbi.delete(txn, key.get());
                } else {
                    dbi.put(txn, key.get(), value);
                }
            }
            txn.commit();
            return oldValue;
        }
    }

    @Override
    public void putAll(List<KeyValue<Bytes, byte[]>> entries) {
        try (Txn<byte[]> txn = env.txnWrite()) {
            for (KeyValue<Bytes, byte[]> entry : entries) {
                Objects.requireNonNull(entry.key, "key cannot be null");
                if (entry.value == null) {
                    dbi.delete(txn, entry.key.get());
                } else {
                    dbi.put(txn, entry.key.get(), entry.value);
                }
            }
            txn.commit();
        }
    }

    @Override
    public StreamObserver<KeyValue<Bytes, byte[]>> putAll() {
        Txn<byte[]> txn = env.txnWrite();
        return new StreamObserver<KeyValue<Bytes, byte[]>>() {
            @Override
            public void onNext(KeyValue<Bytes, byte[]> entry) {
                Objects.requireNonNull(entry.key, "key cannot be null");
                if (entry.value == null) {
                    dbi.delete(txn, entry.key.get());
                } else {
                    dbi.put(txn, entry.key.get(), entry.value);
                }
            }

            @Override
            public void onError(Throwable throwable) {
                txn.abort();
                txn.close();
                LOGGER.error(errorUtils.toGrpcRuntimeException(throwable));
                LOGGER.info("[STORAGESERVICE][PUTALL] error");
            }

            @Override
            public void onCompleted() {
                synchronized (this) {
                    txn.commit();
                    txn.close();
                }
                //env.sync(true);
                LOGGER.info("[STORAGESERVICE][STORE][PUTALL] completed. store info: {}");
            }
        };
    }

    @Override
    public byte[] delete(Bytes key) {
        Objects.requireNonNull(key, "key cannot be null");
        try (Txn<byte[]> txn = env.txnWrite()) {
            byte[] value = dbi.get(txn, key.get());
            dbi.delete(txn, key.get());
            txn.commit();
            return value;
        }
    }

    @Override
    public byte[] get(Bytes key) {
        Objects.requireNonNull(key, "key cannot be null");
        try (Txn<byte[]> txn = env.txnRead()) {
            return dbi.get(txn, key.get());
        }
    }

    @Override
    public KeyValueIterator<Bytes, byte[]> range(Bytes from, Bytes to) {
        validateStoreOpen();
        final LMDBRangeIterator lmdbRangeIterator = new LMDBRangeIterator(from, to);
        openIterators.add(lmdbRangeIterator);
        return lmdbRangeIterator;
    }

    @Override
    public KeyValueIterator<Bytes, byte[]> all() {
        validateStoreOpen();
        final LMDBRangeIterator lmdbIterator = new LMDBRangeIterator(null, null);
        openIterators.add(lmdbIterator);
        return lmdbIterator;
    }

    @Override
    public void destroy() {
        try {
            try (Txn<byte[]> txn = env.txnWrite()) {
                dbi.drop(txn, true);
            }
            String[] files = dbDir.list();
            if (null != files) {
                for (String s : files) {
                    File currentFile = new File(dbDir.getPath(), s);
                    FileUtils.deleteQuietly(currentFile);
                }
            }
        } catch (Exception e) {
            LOGGER.info("[STORAGE] error in destroy: " + ExceptionUtils.getStackTrace(e));
            Thread.currentThread().interrupt();
            throw new RuntimeException(e);
        }
        this.close();
    }


    @Override
    public String name() {
        return storeInfo.getTableName();
    }

    @Override
    public long count() {
        return env.stat().entries;
    }

    private void validateStoreOpen() {
        if (!open) {
            throw new InvalidStateStoreException("Store " + storeInfo + " is currently closed");
        }
    }

    @Override
    public void init(Properties properties) {

        Path dbPath = Paths.get(properties.getProperty(DATA_DIR),
                storeInfo.getNameSpace(),
                storeInfo.getTableName(),
                storeInfo.getFragment() + "");

        try {
            try {
                Files.createDirectories(dbPath);
                dbDir = dbPath.toFile();
                env = Env.create(new ByteArrayProxy()).setMaxDbs(1).setMaxReaders(256).setMapSize(1_073_741_824).open(dbDir, EnvFlags.MDB_NOTLS, EnvFlags.MDB_NOSYNC, EnvFlags.MDB_NOLOCK);
                dbi = env.openDbi((String) null, DbiFlags.MDB_CREATE);
            } catch (final DBException e) {
                throw new ProcessorStateException("Error opening store " + storeInfo + " at location " + dbDir.toString(), e);
            }
        } catch (final IOException e) {
            throw new ProcessorStateException(e);
        }

        open = true;
    }

    @Override
    public void flush() {

    }

    @Override
    public void close() {
        if (!open) {
            return;
        }

        this.open = false;
        closeOpenIterators();
        env.close();
        dbi.close();
    }

    private void closeOpenIterators() {
        final HashSet<KeyValueIterator> iterators;
        synchronized (openIterators) {
            iterators = new HashSet<>(openIterators);
        }
        for (final KeyValueIterator iterator : iterators) {
            iterator.close();
        }
    }

    @Override
    public boolean persistent() {
        return true;
    }

    @Override
    public boolean isOpen() {
        return this.open;
    }

    @Override
    public String toString() {
        return String.format(TOSTRING_FORMAT, storeInfo.toString());
    }

    private class LMDBRangeIterator extends AbstractIterator<KeyValue<Bytes, byte[]>> implements KeyValueIterator<Bytes, byte[]> {
        final CursorIterator<byte[]> cursorIterator;
        final Txn<byte[]> txn;

        private volatile boolean open = true;
        private KeyValue<Bytes, byte[]> next;

        LMDBRangeIterator(final Bytes from, final Bytes to) {
            KeyRange<byte[]> keyRange;
            if (from != null && to != null) {
                keyRange = KeyRange.open(from.get(), to.get());
            } else if (from != null) {
                keyRange = KeyRange.greaterThan(from.get());
            } else if (to != null) {
                keyRange = KeyRange.lessThan(to.get());
            } else {
                keyRange = KeyRange.all();
            }
            this.txn = env.txnRead();
            this.cursorIterator = dbi.iterate(txn, keyRange, Bytes.BYTES_LEXICO_COMPARATOR);
        }

        @Override
        public synchronized void close() {
            openIterators.remove(this);
            txn.close();
            cursorIterator.close();
            this.open = false;
        }

        @Override
        public synchronized boolean hasNext() {
            if (!open) {
                throw new InvalidStateStoreException(String.format("LMDB store %s has closed", storeInfo.toString()));
            }
            return super.hasNext();
        }

        @Override
        public Bytes peekNextKey() {
            if (!hasNext()) {
                throw new NoSuchElementException();
            }
            return next.key;
        }

        @Override
        public KeyValue<Bytes, byte[]> allDone() {
            KeyValue<Bytes, byte[]> rtn = super.allDone();
            close();
            return rtn;
        }

        @Override
        public synchronized KeyValue<Bytes, byte[]> next() {
            return super.next();
        }

        @Override
        protected KeyValue<Bytes, byte[]> makeNext() {
            if (!cursorIterator.hasNext()) {
                return allDone();
            } else {
                CursorIterator.KeyVal<byte[]> keyVal = cursorIterator.next();
                next = new KeyValue<>(Bytes.wrap(keyVal.key()), keyVal.val());
                return next;
            }
        }
    }


}