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

#include "LMDBServicer.h"


using std::cout;
using std::endl;
using std::string;

LMDBServicer::LMDBServicer() : LMDBServicer::LMDBServicer("/tmp") {}

LMDBServicer::LMDBServicer(std::string dataDir) {
    this->dataDir = dataDir;
}

LMDBServicer::~LMDBServicer() {

}

Status defaultStatus = Status::OK;

// put an entry to table
Status LMDBServicer::put(ServerContext *context, const Operand *request, Empty *response) {
    std::exception_ptr eptr;
    try {
        LOG(INFO) << "put request" << endl;
        LMDBStore lmdbStore = getStore(context);

        lmdbStore.put(request);

        LOG(INFO) << "put finished" << endl;
    } catch (...) {
        eptr = std::current_exception();
    }
    handle_eptr(eptr, __FILE__, __LINE__, "LMDBServicer::put");

    return Status::OK;
}

// put an entry to table if absent
Status LMDBServicer::putIfAbsent(ServerContext *context, const Operand *request, Operand *response) {
    std::exception_ptr eptr;
    try {
        LOG(INFO) << "putIfAbsent request" << endl;
        cout << "putIfAbsent request" << endl;
        LMDBStore lmdbStore = getStore(context);

        string_view oldValue = lmdbStore.putIfAbsent(request);

        response->set_key(request->key());
        response->set_value(oldValue.data(), oldValue.size());

        LOG(INFO) << "putIfAbsent finished" << endl;
        cout << "putIfAbsent finished" << endl;
    } catch (...) {
        eptr = std::current_exception();
    }
    handle_eptr(eptr, __FILE__, __LINE__, "LMDBServicer::putIfAbsent");

    return Status::OK;
}

// put entries to table (entries will be streaming in)
Status LMDBServicer::putAll(ServerContext *context, ServerReader<Operand> *reader, Empty *response) {
    std::exception_ptr eptr;
    try {
        LOG(INFO) << "putAll request" << endl;
        cout << "putAll request" << endl;
        LMDBStore lmdbStore = getStore(context);

        lmdbStore.putAll(reader);

        LOG(INFO) << "putAllFinished" << endl;
        cout << "putAllFinished" << endl;
    } catch (...) {
        eptr = std::current_exception();
    }
    handle_eptr(eptr, __FILE__, __LINE__, "LMDBServicer::putAll");

    return Status::OK;
}

// delete an entry from table
Status LMDBServicer::delOne(ServerContext *context, const Operand *request, Operand *response) {
    std::exception_ptr eptr;
    try {
        LOG(INFO) << "delOne request" << endl;
        cout << "delOne request" << endl;
        LMDBStore lmdbStore = getStore(context);

        string_view oldValue = lmdbStore.delOne(request);

        response->set_key(request->key());
        response->set_value(oldValue.data(), oldValue.size());

        LOG(INFO) << "delOne finished" << endl;
        cout << "delOne finished" << endl;
    } catch (...) {
        eptr = std::current_exception();
    }
    handle_eptr(eptr, __FILE__, __LINE__, "LMDBServicer::delOne");

    return Status::OK;
}

// get an entry from table
Status LMDBServicer::get(ServerContext *context, const Operand *request, Operand *response) {
    std::exception_ptr eptr;
    try {
        LOG(INFO) << "get request" << endl;
        cout << "get request" << endl;

        LMDBStore lmdbStore = getStore(context);

        string_view value = lmdbStore.get(request);

        response->set_key(request->key());
        response->set_value(value.data(), value.size());

        LOG(INFO) << "get finished" << endl;
        cout << "get finished" << endl;
    } catch (...) {
        eptr = std::current_exception();
    }
    handle_eptr(eptr, __FILE__, __LINE__, "LMDBServicer::get");

    return Status::OK;
}

// iterate through a table. Response entries are ordered
Status LMDBServicer::iterate(ServerContext *context, const Range *request, ServerWriter<Operand> *writer) {
    std::exception_ptr eptr;
    try {
        LOG(INFO) << "iterate request" << endl;
        cout << "iterate request" << endl;
        LMDBStore lmdbStore = getStore(context);

        lmdbStore.iterate(request, writer);

        LOG(INFO) << "iterate finished" << endl;
        cout << "iterate finished" << endl;
    } catch (...) {
        eptr = std::current_exception();
    }
    handle_eptr(eptr, __FILE__, __LINE__, "LMDBServicer::iterate");

    return Status::OK;
}

// destroy a table
Status LMDBServicer::destroy(ServerContext *context, const Empty *request, Empty *response) {
    std::exception_ptr eptr;
    try {
        LOG(INFO) << "destroy request" << endl;
        cout << "destroy request" << endl;
        LMDBStore lmdbStore = getStore(context);

        lmdbStore.destroy();

        LOG(INFO) << "destroy finished" << endl;
        cout << "destroy finished" << endl;
    } catch (...) {
        eptr = std::current_exception();
    }
    handle_eptr(eptr, __FILE__, __LINE__, "LMDBServicer::destroy");

    return Status::OK;
}

Status LMDBServicer::destroyAll(ServerContext *context, const Empty *request, Empty *response) {
    std::exception_ptr eptr;
    try {
        LOG(INFO) << "destroyAll request" << endl;
        cout << "destroyAll request" << endl;
    } catch (...) {
        eptr = std::current_exception();
    }
    handle_eptr(eptr, __FILE__, __LINE__, "LMDBServicer::destroyAll");

    return Status::OK;
}

// count record amount of a table
Status LMDBServicer::count(ServerContext *context, const Empty *request, Count *response) {
    std::exception_ptr eptr;
    try {
        LOG(INFO) << "count request" << endl;
        cout << "count request" << endl;
        LMDBStore lmdbStore = getStore(context);

        response->set_value(lmdbStore.count());

        LOG(INFO) << "count finished" << endl;
        cout << "count finished" << endl;
    } catch (...) {
        eptr = std::current_exception();
    }
    handle_eptr(eptr, __FILE__, __LINE__, "LMDBServicer::count");

    return Status::OK;
}

Status LMDBServicer::createIfAbsent(ServerContext *context, const CreateTableInfo *request, CreateTableInfo *response) {
    std::exception_ptr eptr;
    try {
        LOG(INFO) << "createIfAbsent request" << endl;
    } catch (...) {
        eptr = std::current_exception();
    }
    handle_eptr(eptr, __FILE__, __LINE__, "LMDBServicer::createIfAbsent");

    return Status::OK;
}

void LMDBServicer::sayHello() {
    LOG(INFO) << "saying hello" << endl;
}

LMDBStore LMDBServicer::getStore(ServerContext *context) {
    LMDBStore lmdbStore;
    std::exception_ptr eptr;
    try {
        std::multimap <grpc::string_ref, grpc::string_ref> metadata = context->client_metadata();

        StoreInfo storeInfo(context);

        bool result = lmdbStore.init(dataDir, storeInfo);

        // todo: add exception if result is false
        if (!result) {
            std::string errorMsg = "Unable to init LMDBStore. please check error log.";
            throw std::runtime_error(errorMsg);
        }
    } catch (...) {
        eptr = std::current_exception();
    }
    handle_eptr(eptr, __FILE__, __LINE__, "LMDBServicer::getStore");

    return lmdbStore;
}