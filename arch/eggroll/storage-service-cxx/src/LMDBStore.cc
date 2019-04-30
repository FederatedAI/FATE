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

#include "LMDBStore.h"

using std::cout;
using std::endl;

LMDBStore::LMDBStore() {
}

LMDBStore::LMDBStore(const LMDBStore& other) {
    this->dbDir = other.dbDir;
    this->storeInfo = other.storeInfo;
}

LMDBStore::~LMDBStore() {
    this->env.close();
}

bool LMDBStore::init(string dataDir, StoreInfo storeInfo) {
    std::stringstream ss;
    string delimiter = "/";
    bool result = true;
    std::exception_ptr eptr;
    try {
        string storeType = LMDB_TEMPORARY;
        if (boost::iequals(storeInfo.getStoreType(), LMDB)) {
            storeType = LMDB;
        }

        ss << dataDir
           << delimiter << storeType
           << delimiter << storeInfo.getNameSpace()
           << delimiter << storeInfo.getTableName()
           << delimiter << storeInfo.getFragment() << delimiter;

        ss >> this->dbDir;

        cout << "dbDir: " << dbDir << endl;

        boost::filesystem::path dst = this->dbDir;
        boost::filesystem::create_directories(dst);

        cout << "ready to open" << endl;

        this->env.set_max_dbs(1).set_max_readers(256).set_mapsize(1UL * 1024UL * 1024UL * 1024UL);

        cout << "env set" << endl;
        this->env.open(dbDir.c_str(), 0, 0644);
    } catch (...) {
        eptr = std::current_exception();
        result = false;
    }
    handle_eptr(eptr, __FILE__, __LINE__, this->toString());

    cout << "inited" << endl;
    return result;
}

void LMDBStore::put(const Operand *operand) {
    lmdb::txn wtxn = lmdb::txn::begin(this->env);

    std::exception_ptr eptr;
    try {
        lmdb::dbi dbi = lmdb::dbi::open(wtxn, nullptr);
        dbi.put(wtxn, operand->key(), operand->value());
    } catch (...) {
        eptr = std::current_exception();
    }
    handle_eptr(eptr, __FILE__, __LINE__, this->toString());

    wtxn.commit();
    wtxn.abort();
}

void LMDBStore::putAll(ServerReader<Operand> *reader) {
    lmdb::txn wtxn = lmdb::txn::begin(this->env);

    std::exception_ptr eptr;
    try {
        lmdb::dbi dbi = lmdb::dbi::open(wtxn, nullptr);
        Operand operand;
        long i = 0;

        long countInterval = 100000;
        long countRemaining = countInterval;

        while (reader->Read(&operand)) {
            dbi.put(wtxn, operand.key(), operand.value());
/*            cout << "key: " << operand.key().c_str() << ", size: " << operand.key().size() << ", strlen: " << strlen(operand.key().c_str())
                << "; value: " << operand.value().c_str() << ", size: " << operand.value().size() << ", strlen: " << strlen(operand.key().c_str()) << endl;*/
            ++i;

            if (--countRemaining == 0) {
                cout << "batch count: " << i << ", at " << std::time(0) << endl;
                countRemaining = countInterval;
            }
        }

        cout << "total putAll: " << i << endl;
        LOG(INFO) << "total putAll: " << i << endl;
    } catch (...) {
        eptr = std::current_exception();
    }

    handle_eptr(eptr, __FILE__, __LINE__, this->toString());

    //iterateAll();
    wtxn.commit();
    wtxn.abort();
}

string_view LMDBStore::putIfAbsent(const Operand *operand) {
    string_view oldValue;
    std::exception_ptr eptr;
    try {
        oldValue = get(operand);
        if (oldValue.empty()) {
            if (operand->value().empty()) {
                delOne(operand);
            } else {
                put(operand);
            }
        }
    } catch (...) {
        eptr = std::current_exception();
    }
    handle_eptr(eptr, __FILE__, __LINE__, this->toString());

    return oldValue;
}

string_view LMDBStore::delOne(const Operand *operand) {
    lmdb::txn wtxn = lmdb::txn::begin(this->env);
    string_view oldValue;

    std::exception_ptr eptr;
    try {
        lmdb::dbi dbi = lmdb::dbi::open(wtxn, nullptr);
        oldValue = get(operand);
        lmdb::val key{operand->key()};

        cout << "----------" << endl
             << "key to delete: " << key.data() << endl;

        bool result = dbi.del(wtxn, key);
        cout << "after delete: result: " << result << endl;
    } catch (...) {
        eptr = std::current_exception();
    }
    handle_eptr(eptr, __FILE__, __LINE__, this->toString());

    wtxn.commit();
    wtxn.abort();

    return oldValue;
}

bool LMDBStore::destroy() {
    bool result = false;
    std::exception_ptr eptr;
    size_t n;
    try {
        n = std::count(dbDir.begin(), dbDir.end(), '/');
        std::stringstream ss;
        string tableName;
        ss << this->storeInfo.getTableName();
        ss >> tableName;
        size_t tableNamePos = dbDir.rfind(tableName);
        cout << "dbDir: " << this->dbDir
        << ", storeInfo: " << this->storeInfo.toString()
        << ", tableName: " << tableName
        << ", tableNamePos: " << tableNamePos
        << ", size: " << dbDir.size() << endl;

        if (n >= 4 && dbDir.substr(0, 4) != "////") {
            //string dirToRemove = dbDir.substr(0, tableNamePos);
            string dirToRemove = dbDir + "/../..";
            cout << "dirToRemove: " << dirToRemove << endl;
            if (boost::filesystem::exists(dirToRemove) && boost::filesystem::is_directory(dirToRemove)) {
                boost::filesystem::remove_all(dirToRemove);
                result = true;
            }
        }
    } catch (...) {
        eptr = std::current_exception();
    }
    handle_eptr(eptr, __FILE__, __LINE__, this->toString());

    cout << "destroy: delimiter count: " << n << ", result: " << result << endl;

    return result;
}

long LMDBStore::count() {
    long result;
    std::exception_ptr eptr;
    try {
        MDB_stat stat;
        lmdb::env_stat(this->env, &stat);
        result = stat.ms_entries;

        //iterateAll();
        cout << "count: " << result << endl;
        LOG(INFO) << "count: " << result << endl;
    } catch (...) {
        eptr = std::current_exception();
    }
    handle_eptr(eptr, __FILE__, __LINE__, this->toString());

    return result;
}

string_view LMDBStore::get(const Operand *operand) {
    lmdb::txn rtxn = lmdb::txn::begin(this->env, nullptr, MDB_RDONLY);
    std::exception_ptr eptr;
    string_view result;
    try {
        lmdb::dbi dbi = lmdb::dbi::open(rtxn, nullptr);

        lmdb::val key{operand->key()};
        lmdb::val value;

        bool found = dbi.get(rtxn, key, value);
        cout << "found: " << found << endl;
        if (found) {
            result = value.to_string_view();
        }
    } catch (...) {
        eptr = std::current_exception();
    }

    handle_eptr(eptr, __FILE__, __LINE__, this->toString());
//    iterateAll();
    rtxn.abort();

    return result;
}

// (a, b]
void LMDBStore::iterate(const Range *range, ServerWriter<Operand> *writer) {
    lmdb::txn rtxn = lmdb::txn::begin(this->env, nullptr, MDB_RDONLY);
    lmdb::dbi dbi = lmdb::dbi::open(rtxn, nullptr);
    lmdb::cursor cursor = lmdb::cursor::open(rtxn, dbi);

    std::exception_ptr eptr;
    try {
        long bytesCount = 0;

        string start = range->start();
        string end = range->end();
        long threshold = range->minchunksize() > 0 ? range->minchunksize() : PAYLOAD_THREASHOLD;
        cout << "start: " << start << " (" << start.size() << "), end: " << end << ", threshold: " << threshold << endl;

        lmdb::val key{start};
        lmdb::val value;

        bool locateStart = true;
        if (start.empty()) {
            locateStart = false;
        }

        cout << "locateStart: " << locateStart << start.empty() << start.size() << endl;
        bool checkEnd = true;
        if (end.empty()) {
            checkEnd = false;
        }

        int count = 0;
        Operand operand;
        string_view curKey;
        size_t keySize;
        size_t valueSize;

        if (locateStart) {
            bool isStartFound = cursor.get(key, value, MDB_SET_RANGE);
            string_view actualStart(key.to_string_view());

            cout << "start: " << start <<
                 ", actual start: " << actualStart
                 << ", isStartFound: " << isStartFound << endl;

            // first element
            if (actualStart != start) {
                keySize = key.size();
                curKey = string_view(key.data(), keySize);
                if (checkEnd && curKey.compare(end) > 0) {
                    return;
                }
                operand.set_key(key.data(), keySize);

                valueSize = value.size();
                operand.set_value(value.data(), valueSize);
                writer->Write(operand);
                ++count;

                bytesCount += keySize + valueSize;

/*                cout << "key size: " << key.size() << ", key data: " << curKey << ", value size: " << value.size()
                     << ", value data: " << value.to_string_view() << endl;*/
            }
        }

        cout << "located start: " << key.to_string() << endl;

        // 2 to last
        while (cursor.get(key, value, MDB_NEXT)) {
            if (bytesCount >= threshold) {
                break;
            }

            keySize = key.size();
            curKey = string_view(key.data(), keySize);

            if (checkEnd && curKey.compare(end) > 0) {
                cout << "breaking. curKey: " << curKey << ", end: " << end << endl;
                break;
            }

            operand.set_key(key.data(), keySize);

            valueSize = value.size();
            operand.set_value(value.data(), valueSize);

            writer->Write(operand);

/*            cout << "key size: " << key.size() << ", key data: " << curKey << ", value size: " << value.size()
                 << ", value data: " << value.to_string() << endl;*/

            ++count;
            bytesCount += keySize + valueSize;
        }
        cout << "total iterated: " << count << endl;
        LOG(INFO) << "total iterated: " << count << endl;
    } catch (...) {
        eptr = std::current_exception();
    }

    handle_eptr(eptr, __FILE__, __LINE__, this->toString());

    cursor.close();
    rtxn.abort();
}

void LMDBStore::iterateAll() {
    lmdb::txn rtxn = lmdb::txn::begin(this->env, nullptr, MDB_RDONLY);
    lmdb::dbi dbi = lmdb::dbi::open(rtxn, nullptr);
    lmdb::cursor cursor = lmdb::cursor::open(rtxn, dbi);

    std::exception_ptr eptr;
    try {
        std::string key, value;

        int total = 0;

        while (cursor.get(key, value, MDB_NEXT)) {
            std::printf("key: '%s' (%lu), value: '%s' (%lu)\n", key.c_str(), key.size(), value.c_str(), value.size());
            ++total;
        }
        cout << "iterateAll total: " << total << endl;
    } catch (...) {
        eptr = std::current_exception();
    }
    handle_eptr(eptr, __FILE__, __LINE__, this->toString());

    cursor.close();
    rtxn.abort();
}

string LMDBStore::toString() {
    std::stringstream ss;
    std::exception_ptr eptr;
    try {
        ss << "{storeInfo: " << this->storeInfo.toString()
           << ", dbDir: " << this->dbDir
           << "}";
    } catch (...) {
        eptr = std::current_exception();
    }
    handle_eptr(eptr, __FILE__, __LINE__, "error in LMDBStore::toString()");

    string result;
    ss >> result;

    return result;
}