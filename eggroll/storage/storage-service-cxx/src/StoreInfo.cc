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

#include "StoreInfo.h"

using std::string;
using grpc::ServerContext;
using grpc::string_ref;

StoreInfo::StoreInfo() {
    
}

StoreInfo::StoreInfo(const StoreInfo& other) {
    this->storeType = other.getStoreType();
    this->fragment = other.getFragment();
    this->tableName = other.getTableName();
    this->nameSpace = other.getNameSpace();
}

StoreInfo::StoreInfo(const ServerContext* context) {
    fromGrpcContext(context);
}

void StoreInfo::fromGrpcContext(const ServerContext* context) {
    if (NULL == context) {
        return;
    }

    std::multimap<grpc::string_ref, grpc::string_ref> metadata = context->client_metadata();

    this->nameSpace = findContextKey(metadata, NAME_SPACE);
    this->tableName = findContextKey(metadata, TABLE_NAME);
    this->storeType = findContextKey(metadata, STORE_TYPE);
    string_ref fragmentStr = findContextKey(metadata, FRAGMENT);
    if (!fragmentStr.empty()) {
        this->fragment = fastAtoi(fragmentStr);
    }
}

string_ref StoreInfo::getStoreType() const {
    return this->storeType;
}

string_ref StoreInfo::getNameSpace() const {
    return this->nameSpace;
}

string_ref StoreInfo::getTableName() const {
    return this->tableName;
}

int StoreInfo::getFragment() const {
    return this->fragment;
}

string_ref StoreInfo::findContextKey(std::multimap<grpc::string_ref, grpc::string_ref> metadata, const char * key) {
    auto target = metadata.find(key);
    if (target == metadata.end()) {
        return "";
    } else {
        return target->second;
    }
}

int StoreInfo::fastAtoi(const char * str) {
    int val = 0;
    while (*str) {
        val = val * 10 + (*str++ - '0');
    }
    return val;
}

int StoreInfo::fastAtoi(string_ref& str) {
    int val = 0;
    for (auto iter = str.begin(); iter != str.end(); ++iter) {
        val = val * 10 + (*iter - '0');
    }

    return val;
}

string StoreInfo::toString() {
    std::stringstream ss;
    ss << "{nameSpace: " << this->nameSpace
    << ", tableName: " << this->tableName
    << ", fragment: " << this->fragment
    << ", storeType: " << this->storeType
    << "}";

    string result;

    ss >> result;

    return result;
}
