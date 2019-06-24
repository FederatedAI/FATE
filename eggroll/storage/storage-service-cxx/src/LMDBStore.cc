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
    this->_dbDir = other._dbDir;
    this->storeInfo = other.storeInfo;
}

LMDBStore::~LMDBStore() {
    LOG(INFO) << "[LMDBStore::~LMDBStore] dbDir: " << _dbDir << ", desctructor use count: " << _env.use_count() << endl;
}

bool LMDBStore::init(string dataDir, StoreInfo& storeInfo) {
    std::stringstream ss;
    string delimiter = "/";
    bool result = true;
    std::exception_ptr eptr;
    try {
        string storeType = LMDB_TEMPORARY;
        if (boost::iequals(storeInfo.getStoreType(), LMDB)) {
            storeType = LMDB;
        }

        cout << storeInfo.toString() << endl;

        ss << dataDir
           << delimiter << storeType
           << delimiter << storeInfo.getNameSpace()
           << delimiter << storeInfo.getTableName()
           << delimiter << storeInfo.getFragment() << delimiter;

        ss >> this->_dbDir;

        cout << "[LMDBStore::init] dbDir: " << _dbDir << endl;
        LOG(INFO) << "[LMDBStore::init] dbDir: " << _dbDir << endl;

        boost::filesystem::path dst = this->_dbDir;
        boost::filesystem::create_directories(dst);

        cout << "[LMDBStore::init] ready to open: " << _dbDir << endl;

//        this->env.set_max_dbs(1).set_max_readers(256).set_mapsize(1UL * 1024UL * 1024UL * 1024UL);

        cout << "[LMDBStore::init] env set: " << _dbDir << endl;
//        this->env.open(dbDir.c_str(), 0, 0644);

        this->storeInfo = storeInfo;
        this->_env = getMDBEnv(_dbDir.data(), 0, 0644);
        this->_dbi = this->_env->openDB("main", MDB_CREATE);
        LOG(INFO) << "[LMDBStore::init] inited. dbdir: " << _dbDir << ", use_count: " << _env.use_count() << endl;
        cout << "[LMDBStore::init] inited. dbdir: " << _dbDir << ", use_count: " << _env.use_count() << endl;
    } catch (...) {
        eptr = std::current_exception();
        result = false;
    }
    handle_eptr(eptr, __FILE__, __LINE__, this->toString());

    return result;
}

void LMDBStore::put(const Operand *operand) {
    LOG(INFO) << "[LMDBStore::put] dbDir: " << _dbDir << endl;
    MDBRWTransaction rwtxn = _env->getRWTransaction();
    std::exception_ptr eptr;
    try {
        rwtxn.put(_dbi, operand->key(), operand->value());
        rwtxn.commit();
    } catch (...) {
        eptr = std::current_exception();
        rwtxn.abort();
    }
    handle_eptr(eptr, __FILE__, __LINE__, this->toString());
}

long LMDBStore::putAll(ServerReader<Operand> *reader) {
    LOG(INFO) << "[LMDBStore::putAll] dbDir: " << _dbDir << endl;
    MDBRWTransaction rwtxn = _env->getRWTransaction();
    long i = 0;

    std::exception_ptr eptr;
    try {
        Operand operand;

        long countInterval = 100000;
        long countRemaining = countInterval;

        while (reader->Read(&operand)) {
            rwtxn.put(_dbi, operand.key(), operand.value());
            /*
            cout << "key: " << operand.key().c_str() << ", size: " << operand.key().size() << ", strlen: " << strlen(operand.key().c_str())
                << "; value: " << operand.value().c_str() << ", size: " << operand.value().size() << ", strlen: " << strlen(operand.key().c_str()) << endl;
                */
            ++i;

            if (--countRemaining == 0) {
                cout << "batch count: " << i << ", at " << std::time(0) << endl;
                countRemaining = countInterval;
            }
        }

        rwtxn.commit();
        cout << "[LMDBStore::putAll] dbDir: " << _dbDir << ", total putAll: " << i << endl;
        LOG(INFO) << "[LMDBStore::putAll] dbDir: " << _dbDir << ", total putAll: " << i << endl;
    } catch (...) {
        eptr = std::current_exception();
        rwtxn.abort();
    }
    handle_eptr(eptr, __FILE__, __LINE__, this->toString());
    return i;
}

string_view LMDBStore::putIfAbsent(const Operand *operand) {
    LOG(INFO) << "[LMDBStore::putIfAbsent] dbDir: " << _dbDir << endl;
    MDBRWTransaction rwtxn = _env->getRWTransaction();
    string_view result;
    std::exception_ptr eptr;
    try {
        string_view key = operand->key();
        int rc = rwtxn.get(_dbi, key, result);
        if (MDB_NOTFOUND == rc) {
            rwtxn.put(_dbi, key, operand->value());
            result = operand->value();
        }
        cout << "putIfAbsent. rc: " << rc << ", result: " << result << endl;
        rwtxn.commit();
    } catch (...) {
        eptr = std::current_exception();
        rwtxn.abort();
    }
    handle_eptr(eptr, __FILE__, __LINE__, this->toString());

    return result;
}

string_view LMDBStore::delOne(const Operand *operand) {
    LOG(INFO) << "[LMDBStore::delOne] dbDir: " << _dbDir << endl;
    MDBRWTransaction rwtxn = _env->getRWTransaction();
    string_view oldValue;

    std::exception_ptr eptr;
    try {
        string_view key = operand->key();
        int rc = rwtxn.get(_dbi, key, oldValue);

        cout << "----------" << endl
             << "key to delete: " << key.data() << endl;

        if (MDB_NOTFOUND != rc) {
            rwtxn.del(_dbi, key);
        }
        rwtxn.commit();
        cout << "after delete: result: " << oldValue << endl;
    } catch (...) {
        eptr = std::current_exception();
        rwtxn.abort();
    }
    handle_eptr(eptr, __FILE__, __LINE__, this->toString());

    return oldValue;
}

bool LMDBStore::destroy() {
    LOG(INFO) << "[LMDBStore::destroy] dbDir: " << _dbDir << endl;
    bool result = false;
    std::exception_ptr eptr;
    size_t n;
    try {
        int env_use_count = _env.use_count();
        if (env_use_count > 1) {
            LOG(INFO) << "unable to destroy " << _dbDir << ". env use_count: " << env_use_count << endl;
            return false;
        }
        n = std::count(_dbDir.begin(), _dbDir.end(), '/');
        std::stringstream ss;
        string tableName;
        ss << this->storeInfo.getTableName();
        ss >> tableName;
        size_t tableNamePos = _dbDir.rfind(tableName);
        cout << "dbDir: " << _dbDir
        << ", storeInfo: " << this->storeInfo.toString()
        << ", tableName: " << tableName
        << ", tableNamePos: " << tableNamePos
        << ", size: " << _dbDir.size() << endl;

        if (n >= 4 && _dbDir.substr(0, 4) != "////") {
            //string dirToRemove = dbDir.substr(0, tableNamePos);
            string dirToRemove = _dbDir.substr(0, tableNamePos) + tableName;
            LOG(INFO) << "[LMDBStore::destroy] dirToRemove: " << dirToRemove << endl;
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
    LOG(INFO) << "[LMDBStore::count] dbDir: " << _dbDir << endl;
    long result;
    MDBROTransaction rotxn = _env->getROTransaction();
    std::exception_ptr eptr;
    try {
        MDB_stat stat;
        mdb_stat(rotxn, _dbi, &stat);
        result = stat.ms_entries;

        // iterateAll();
        cout << "count: " << result << endl;
        LOG(INFO) << "count: " << result << endl;
    } catch (...) {
        eptr = std::current_exception();
    }
    handle_eptr(eptr, __FILE__, __LINE__, this->toString());

    return result;
}

string_view LMDBStore::get(const Operand *operand) {
    LOG(INFO) << "[LMDBStore::get] dbDir: " << _dbDir << endl;
    string_view result;
    MDBROTransaction rotxn = _env->getROTransaction();

    std::exception_ptr eptr;
    try {
        int found = rotxn.get(_dbi, operand->key(), result);
        cout << "found: " << found << ", result: " << result << endl;
    } catch (...) {
        eptr = std::current_exception();
    }
    handle_eptr(eptr, __FILE__, __LINE__, this->toString());

    // iterateAll();
    return result;
}

// (a, b]
void LMDBStore::iterate(const Range *range, ServerWriter<Operand> *writer) {
    LOG(INFO) << "[LMDBStore::iterate] dbDir: " << _dbDir << endl;
    MDBROTransaction rotxn = _env->getROTransaction();
    MDBROCursor rocursor = rotxn.getCursor(_dbi);
    int count = 0;

    std::exception_ptr eptr;
    try {
        long bytesCount = 0;

        string start = range->start();
        string end = range->end();
        long threshold = range->minchunksize() > 0 ? range->minchunksize() : PAYLOAD_THREASHOLD;
        cout << "start: " << start << " (" << start.size() << "), end: " << end << ", threshold: " << threshold << endl;

        MDBOutVal key, val;
        string_view keyView, valView;

        bool locateStart = true;
        if (start.empty()) {
            locateStart = false;
        }

        cout << "locateStart: " << locateStart << start.empty() << start.size() << endl;
        bool checkEnd = true;
        if (end.empty()) {
            checkEnd = false;
        }

        int rc;
        Operand operand;

        // first element
        if (locateStart) {
            // key == actual start returned by the call
            rc = rocursor.lower_bound(start, key, val);
            keyView = key.get<string_view>();
            valView = val.get<string_view>();

            cout << "start: " << start <<
                 ", actual start: " << keyView
                 << ", isStartFound: " << (start == keyView);
            cout << endl;

            // first element
            if (keyView != start) {
                if (checkEnd && keyView.compare(end) > 0) {
                    return;
                }
                operand.set_key(keyView.data(), keyView.size());
                operand.set_value(valView.data(), valView.size());
                writer->Write(operand);
                ++count;

                bytesCount += keyView.size() + valView.size();

/*                cout << "key size: " << key.size() << ", key data: " << curKey << ", value size: " << value.size()
                     << ", value data: " << value.to_string_view() << endl;*/
            }
        }

        cout << "located start: " << keyView << endl;

        // 2 to last
        while (0 == rocursor.next(key, val)) {
            if (bytesCount >= threshold) {
                break;
            }

            keyView = key.get<string_view>();
            valView = val.get<string_view>();

            if (checkEnd && keyView.compare(end) > 0) {
                cout << "breaking. curKey: " << keyView << ", end: " << end << endl;
                break;
            }

            operand.set_key(keyView.data(), keyView.size());
            operand.set_value(valView.data(), valView.size());

            writer->Write(operand);

/*            cout << "key size: " << key.size() << ", key data: " << curKey << ", value size: " << value.size()
                 << ", value data: " << value.to_string() << endl;*/

            ++count;
            bytesCount += keyView.size() + valView.size();
        }
        cout << "[LMDBStore::iterate] dbDir: " << _dbDir << ", total iterated: " << count << endl;
        LOG(INFO) << "[LMDBStore::iterate] dbDir: " << _dbDir << ", total iterated: " << count << endl;
    } catch (...) {
        eptr = std::current_exception();
    }

    handle_eptr(eptr, __FILE__, __LINE__, this->toString());
}

void LMDBStore::iterateAll() {
    MDBROTransaction rotxn = _env->getROTransaction();
    MDBROCursor rocursor = rotxn.getCursor(_dbi);

    cout << "------ iterateAll ------" << endl;

    std::exception_ptr eptr;
    try {

        MDBOutVal key, val;

        int count = 0;

          while(!rocursor.get(key, val, count ? MDB_NEXT : MDB_FIRST)) {
            cout << key.get<string>();
            cout<<": " << val.get<string>();
            cout << "\n";
            ++count;

          }

        cout << endl << "------ iterateAll total: " << count << " ------" << endl;

        int rc;
        if (!(rc = rocursor.find("it", key, val))) {
            string_view keyResult(key.get<string_view>());
            cout << keyResult << ": " << val.get<string_view>() << endl;
        } else {
            cout << "rc: " << rc << endl;
        }
    } catch (...) {
        eptr = std::current_exception();
    }
    handle_eptr(eptr, __FILE__, __LINE__, this->toString());
}

string LMDBStore::toString() {
    std::stringstream ss;
    std::exception_ptr eptr;
    try {
        ss << "{storeInfo: " << this->storeInfo.toString()
           << ", dbDir: " << this->_dbDir
           << "}";
    } catch (...) {
        eptr = std::current_exception();
    }
    handle_eptr(eptr, __FILE__, __LINE__, "error in LMDBStore::toString()");

    string result;
    ss >> result;

    return result;
}