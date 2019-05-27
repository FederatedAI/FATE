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

#ifndef STORAGE_SERVICE_CXX_LMDBSERVICER_H
#define STORAGE_SERVICE_CXX_LMDBSERVICER_H

#include <iostream>
#include <memory>
#include <string>
#include <map>
#include <iterator>

#include <grpcpp/grpcpp.h>
#include <grpcpp/server_context.h>
#include <glog/logging.h>
#include <boost/utility/string_view.hpp>

#include "storage.grpc.pb.h"
#include "LMDBStore.h"
#include "StoreInfo.h"
#include "ExceptionHandler.h"

using grpc::Status;
using grpc::ServerContext;
using grpc::ServerReader;
using grpc::ServerWriter;
using namespace com::webank::ai::fate::api::eggroll::storage;

class LMDBServicer final : public KVService::Service {
public:
    LMDBServicer();
    LMDBServicer(std::string dataDir);
    virtual ~LMDBServicer() override ;
    // put an entry to table
    Status put(ServerContext* context, const Operand* request, Empty* response) override;
    // put an entry to table if absent
    Status putIfAbsent(ServerContext* context, const Operand* request, Operand* response) override;
    // put entries to table (entries will be streaming in)
    Status putAll(ServerContext* context, ServerReader<Operand>* reader, Empty* response) override;
    // delete an entry from table
    Status delOne(ServerContext* context, const Operand* request, Operand* response) override;
    // get an entry from table
    Status get(ServerContext* context, const Operand* request, Operand* response) override;
    // iterate through a table. Response entries are ordered
    Status iterate(ServerContext* context, const Range* request, ServerWriter<Operand>* writer) override;
    // destroy a table
    Status destroy(ServerContext* context, const Empty* request, Empty* response) override;
    // destroy multiple tables
    Status destroyAll(ServerContext* context, const Empty* request, Empty* response) override;
    // count record amount of a table
    Status count(ServerContext* context, const Empty* request, Count* response) override;
    // create a table
    Status createIfAbsent(ServerContext* context, const CreateTableInfo* request, CreateTableInfo* response) override;

    void sayHello();

private:
    LMDBStore getStore(ServerContext* context);

    string dataDir;
};


#endif //STORAGE_SERVICE_CXX_LMDBSERVICER_H
