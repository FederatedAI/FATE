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

package com.webank.ai.eggroll.framework.storage.service.model;

import com.webank.ai.eggroll.core.error.exception.InvalidStateStoreException;
import com.webank.ai.eggroll.core.error.exception.ProcessorStateException;
import com.webank.ai.eggroll.core.io.KeyValue;
import com.webank.ai.eggroll.core.io.KeyValueIterator;
import com.webank.ai.eggroll.core.io.KeyValueStore;
import com.webank.ai.eggroll.core.io.StoreInfo;
import com.webank.ai.eggroll.core.model.Bytes;
import com.webank.ai.eggroll.core.utils.AbstractIterator;
import io.grpc.stub.StreamObserver;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.iq80.leveldb.*;
import org.iq80.leveldb.impl.Iq80DBFactory;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

public class LevelDBStore implements KeyValueStore<Bytes, byte[]> {

    public static final String DATA_DIR = "data.dir";
    private static final int WRITE_BUFFER_SIZE = 16 * 1024 * 1024;
    private static final int BLOCK_RESTART_INTERVAL = 16;
    private static final int BLOCK_SIZE = 4096;
    private static final int MAX_OPEN_FILES = 1000;
    private static final CompressionType COMPRESSION_TYPE = CompressionType.NONE;
    private static final String DB_FILE_DIR = "leveldb";
    private static Logger LOGGER = LogManager.getLogger(LevelDBStore.class);
    private final StoreInfo storeInfo;
    private final Set<KeyValueIterator> openIterators = Collections.synchronizedSet(new HashSet<KeyValueIterator>());
    protected volatile boolean open = false;
    File dbDir;
    private Options options;
    private WriteOptions wOptions;
    private ReadOptions rOptions;
    private DB db;


    public LevelDBStore(StoreInfo storeInfo) {
        this.storeInfo = storeInfo;
    }

    @Override
    public void init(Properties properties) {
        options = new Options();
        options.writeBufferSize(WRITE_BUFFER_SIZE);
        options.compressionType(COMPRESSION_TYPE);
        options.createIfMissing(true);
        options.errorIfExists(false);
        options.blockSize(BLOCK_SIZE);
        options.blockRestartInterval(BLOCK_RESTART_INTERVAL);
        options.maxOpenFiles(MAX_OPEN_FILES);

        wOptions = new WriteOptions();

        rOptions = new ReadOptions();

        File dataDir = new File(properties.getProperty(DATA_DIR));

        dbDir = Paths.get(dataDir.getAbsolutePath(), DB_FILE_DIR,
                storeInfo.getNameSpace(),
                storeInfo.getTableName(),
                storeInfo.getFragment() + "")
                .toFile();

        try {
            try {
                Files.createDirectories(dbDir.getParentFile().toPath());
                db = Iq80DBFactory.factory.open(dbDir, options);
            } catch (final DBException e) {
                throw new ProcessorStateException("Error opening store " + storeInfo + " at location " + dbDir.toString(), e);
            }
        } catch (final IOException e) {
            throw new ProcessorStateException(e);
        }

        open = true;
    }

    private void validateStoreOpen() {
        if (!open) {
            throw new InvalidStateStoreException("Store " + storeInfo + " is currently closed");
        }
    }

    private void putInternal(final byte[] rawKey,
                             final byte[] rawValue) {
        if (rawValue == null) {
            try {
                db.delete(rawKey, wOptions);
            } catch (final DBException e) {
                throw new ProcessorStateException("Error while removing key %s from store " + storeInfo, e);
            }
        } else {
            try {
                db.put(rawKey, rawValue, wOptions);
            } catch (final DBException e) {
                throw new ProcessorStateException("Error while putting key %s value %s into store " + storeInfo, e);
            }
        }
    }

    private byte[] getInternal(final byte[] rawKey) {
        try {
            return db.get(rawKey, rOptions);
        } catch (final DBException e) {
            throw new ProcessorStateException("Error while getting value for key %s from store " + storeInfo, e);
        }
    }

    private void write(final WriteBatch batch) throws DBException, IOException {
        try {
            db.write(batch, wOptions);
        } finally {
            batch.close();
        }
    }

    @Override
    public synchronized void put(final Bytes key, final byte[] value) {
        Objects.requireNonNull(key, "key cannot be null");
        validateStoreOpen();
        putInternal(key.get(), value);
    }

    @Override
    public synchronized byte[] putIfAbsent(final Bytes key, final byte[] value) {
        Objects.requireNonNull(key, "key cannot be null");
        final byte[] oldValue = get(key);
        if (oldValue == null) {
            put(key, value);
        }
        return oldValue;
    }

    @Override
    public void putAll(List<KeyValue<Bytes, byte[]>> entries) {
        try (final WriteBatch batch = db.createWriteBatch()) {
            for (final KeyValue<Bytes, byte[]> entry : entries) {
                Objects.requireNonNull(entry.key, "key cannot be null");
                if (entry.value == null) {
                    batch.delete(entry.key.get());
                } else {
                    batch.put(entry.key.get(), entry.value);
                }
            }
            write(batch);
        } catch (IOException e) {
            throw new ProcessorStateException("Error while batch writing to store " + storeInfo, e);
        }
    }

    @Override
    public StreamObserver<KeyValue<Bytes, byte[]>> putAll() {
        return new StreamObserver<KeyValue<Bytes, byte[]>>() {
            final WriteBatch batch = db.createWriteBatch();

            @Override
            public void onNext(KeyValue<Bytes, byte[]> entry) {
                Objects.requireNonNull(entry.key, "key cannot be null");
                if (entry.value == null) {
                    batch.delete(entry.key.get());
                } else {
                    batch.put(entry.key.get(), entry.value);
                }
            }

            @Override
            public void onError(Throwable throwable) {
                LOGGER.error(throwable);
                try {
                    batch.close();
                } catch (IOException e) {
                    LOGGER.error(e);
                }
            }

            @Override
            public void onCompleted() {
                try {
                    write(batch);
                    batch.close();
                } catch (IOException e) {
                    throw new ProcessorStateException("Error while batch writing to store " + storeInfo, e);
                }
            }
        };
    }

    @Override
    public synchronized byte[] delete(Bytes key) {
        Objects.requireNonNull(key, "key cannot be null");
        final byte[] value = get(key);
        put(key, null);
        return value;
    }

    @Override
    public synchronized byte[] get(Bytes key) {
        validateStoreOpen();
        return getInternal(key.get());
    }

    @Override
    public synchronized KeyValueIterator<Bytes, byte[]> range(Bytes from, Bytes to) {
        if (from == null && to == null) {
            return all();
        }
        validateStoreOpen();

        final LevelDBRangeIterator levelDBRangeIterator = new LevelDBRangeIterator(storeInfo.getTableName(), db.iterator(rOptions), from, to);
        openIterators.add(levelDBRangeIterator);

        return levelDBRangeIterator;
    }

    @Override
    public synchronized KeyValueIterator<Bytes, byte[]> all() {
        validateStoreOpen();
        final DBIterator innerIter = db.iterator(rOptions);
        innerIter.seekToFirst();
        final LevelDBIterator levelDBIterator = new LevelDBIterator(storeInfo.getTableName(), innerIter);
        openIterators.add(levelDBIterator);
        return levelDBIterator;
    }

    @Override
    public synchronized void destroy() {
        close();
        try {
            Iq80DBFactory.factory.destroy(dbDir, new Options());
        } catch (IOException e) {
            throw new ProcessorStateException("Error while destroying store " + storeInfo, e);
        }
    }

    @Override
    public long count() {
        throw new UnsupportedOperationException("count operation not supported");
    }

    @Override
    public synchronized void close() {
        if (!open) {
            return;
        }

        open = false;
        closeOpenIterators();

        try {
            db.close();
        } catch (IOException e) {
            // ignore this
        }

        options = null;
        wOptions = null;
        rOptions = null;
        db = null;
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
    public void flush() {

    }


    @Override
    public String name() {
        return this.storeInfo.getTableName();
    }


    @Override
    public boolean isOpen() {
        return this.open;
    }


    private class LevelDBIterator extends AbstractIterator<KeyValue<Bytes, byte[]>> implements KeyValueIterator<Bytes, byte[]> {
        private final String storeName;
        private final DBIterator iter;

        private volatile boolean open = true;

        private KeyValue<Bytes, byte[]> next;

        LevelDBIterator(final String storeName, final DBIterator iter) {
            this.storeName = storeName;
            this.iter = iter;
        }

        @Override
        public synchronized void close() {
            openIterators.remove(this);
            try {
                iter.close();
            } catch (IOException e) {
                // Ignore this
            }
            open = false;
        }

        @Override
        public Bytes peekNextKey() {
            if (!hasNext()) {
                throw new NoSuchElementException();
            }
            return next.key;
        }

        @Override
        public synchronized boolean hasNext() {
            if (!open) {
                throw new InvalidStateStoreException(String.format("LevelDB store %s has closed", storeName));
            }
            return super.hasNext();
        }

        @Override
        public synchronized KeyValue<Bytes, byte[]> next() {
            return super.next();
        }

        @Override
        protected KeyValue<Bytes, byte[]> makeNext() {
            if (!iter.hasNext()) {
                return allDone();
            } else {
                next = getKeyValue();
                return next;
            }
        }

        private KeyValue<Bytes, byte[]> getKeyValue() {
            final Map.Entry<byte[], byte[]> entry = iter.next();
            return new KeyValue<>(new Bytes(entry.getKey()), entry.getValue());
        }
    }

    private class LevelDBRangeIterator extends LevelDBIterator {
        private final Comparator<byte[]> comparator = Bytes.BYTES_LEXICO_COMPARATOR;
        private final byte[] rawToKey;

        LevelDBRangeIterator(final String storeName,
                             final DBIterator iter,
                             final Bytes from,
                             final Bytes to) {
            super(storeName, iter);
            if (from != null) {
                iter.seek(from.get());
                if (iter.hasNext() && comparator.compare(iter.peekNext().getKey(), from.get()) == 0) {
                    iter.next();
                }
            }
            if (to != null)
                rawToKey = to.get();
            else
                rawToKey = null;
        }

        @Override
        public KeyValue<Bytes, byte[]> makeNext() {
            final KeyValue<Bytes, byte[]> next = super.makeNext();

            if (next == null) {
                return allDone();
            } else {
                if (rawToKey == null || comparator.compare(next.key.get(), rawToKey) < 0)
                    return next;
                else
                    return allDone();
            }
        }
    }


}
