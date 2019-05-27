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

#ifndef STORAGE_SERVICE_CXX_STOREINFO_H
#define STORAGE_SERVICE_CXX_STOREINFO_H

#include <string>
#include <map>
#include <memory>
#include <iterator>
#include <sstream>

#include <grpcpp/grpcpp.h>

#include "constants.h"

class StoreInfo {
public:
    StoreInfo();
    StoreInfo(const StoreInfo& other);
    StoreInfo(const grpc::ServerContext* context);
    void fromGrpcContext(const grpc::ServerContext* context);
    grpc::string_ref getStoreType() const;
    grpc::string_ref getNameSpace() const;
    grpc::string_ref getTableName() const;
    int getFragment() const;
    std::string toString();
private:
    grpc::string_ref findContextKey(std::multimap<grpc::string_ref, grpc::string_ref> metadata, const char * key);
    int fastAtoi(const char * str);
    int fastAtoi(grpc::string_ref& str);

    grpc::string_ref storeType;
    grpc::string_ref nameSpace;
    grpc::string_ref tableName;
    int fragment = -1;
};

#endif //STORAGE_SERVICE_CXX_STOREINFO_H
