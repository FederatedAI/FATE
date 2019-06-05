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
    cout << "desctructor use count: " << _env.use_count() << endl;
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

//        this->env.set_max_dbs(1).set_max_readers(256).set_mapsize(1UL * 1024UL * 1024UL * 1024UL);

        cout << "env set" << endl;
//        this->env.open(dbDir.c_str(), 0, 0644);

        this->_env = getMDBEnv(dbDir.c_str(), 0, 0644);
        this->_dbi = this->_env->openDB(dbDir, MDB_CREATE);
    } catch (...) {
        eptr = std::current_exception();
        result = false;
    }
    handle_eptr(eptr, __FILE__, __LINE__, this->toString());

    cout << "inited. use_count: " << _env.use_count() << endl;
    return result;
}

void LMDBStore::put(const Operand *operand) {
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
        cout << "total putAll: " << i << endl;
        LOG(INFO) << "total putAll: " << i << endl;
    } catch (...) {
        eptr = std::current_exception();
        rwtxn.abort();
    }
    handle_eptr(eptr, __FILE__, __LINE__, this->toString());
    return i;
}

string_view LMDBStore::putIfAbsent(const Operand *operand) {
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
    bool result = false;
    MDBRWTransaction rwtxn = _env->getRWTransaction();

    std::exception_ptr eptr;
    size_t n;
    try {
        int env_use_count = _env.use_count();
        if (env_use_count > 1) {
            LOG(INFO) << "unable to destroy " << dbDir << ". env use_count: " << env_use_count << endl;
            rwtxn.abort();
            return false;
        }
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

        mdb_drop(rwtxn, _dbi, 1);
        rwtxn.commit();
        if (n >= 4 && dbDir.substr(0, 4) != "////") {
            //string dirToRemove = dbDir.substr(0, tableNamePos);
            string dirToRemove = dbDir + "/../..";
            LOG(INFO) << "dirToRemove: " << dirToRemove << endl;
            cout << "dirToRemove: " << dirToRemove << endl;
            if (boost::filesystem::exists(dirToRemove) && boost::filesystem::is_directory(dirToRemove)) {
                boost::filesystem::remove_all(dirToRemove);
                result = true;
            }
        }
    } catch (...) {
        eptr = std::current_exception();
        rwtxn.abort();
    }
    handle_eptr(eptr, __FILE__, __LINE__, this->toString());

    cout << "destroy: delimiter count: " << n << ", result: " << result << endl;

    return result;
}

long LMDBStore::count() {
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
    MDBROTransaction rotxn = _env->getROTransaction();
    MDBROCursor rocursor = rotxn.getCursor(_dbi);

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
        int count = 0;
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
        cout << "total iterated: " << count << endl;
        LOG(INFO) << "total iterated: " << count << endl;
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