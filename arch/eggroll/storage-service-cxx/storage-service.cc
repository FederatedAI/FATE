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

#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <sstream>

#include <grpcpp/grpcpp.h>
#include <glog/logging.h>

#include "src/LMDBServicer.h"
#include "src/LMDBStore.h"
#include "src/ExceptionHandler.h"

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;

using std::string;
using std::cout;
using std::endl;

void RunServer(int port, string dataDir) {
    std::exception_ptr eptr;
    try {
        std::stringstream ss;
        string serverAddress;

        ss << "0.0.0.0:" << port;
        serverAddress = ss.str();

        LMDBServicer lmdbServicer(dataDir);

        ServerBuilder builder;
        builder.AddListeningPort(serverAddress, grpc::InsecureServerCredentials());
        builder.RegisterService(&lmdbServicer);

        std::unique_ptr <Server> server(builder.BuildAndStart());
        cout << "Server listening on " << serverAddress << ", dataDir: " << dataDir << endl;
        LOG(INFO) << "Server listening on " << serverAddress << ", dataDir: " << dataDir;
        server->Wait();
    } catch (...) {
        eptr = std::current_exception();
    }
    handle_eptr(eptr, __FILE__, __LINE__, "RunServer");
}

class InputParser {
public:
    InputParser(int &argc, char **argv) {
        for (int i = 1; i < argc; ++i) {
            this->tokens.push_back(std::string(argv[i]));
        }
    }

    const std::string &getCmdOption(const std::string &option) const {
        std::vector<std::string>::const_iterator itr;
        itr = std::find(this->tokens.begin(), this->tokens.end(), option);
        if (itr != this->tokens.end() && ++itr != this->tokens.end()) {
            return *itr;
        }
        static const std::string empty_string("");
        return empty_string;
    }

    bool cmdOptionExists(const std::string &option) const {
        return std::find(this->tokens.begin(), this->tokens.end(), option)
               != this->tokens.end();
    }

private:
    std::vector <std::string> tokens;
};

int main(int argc, char **argv) {
    InputParser inputParser(argc, argv);

    int port = -1;

    if (!inputParser.cmdOptionExists("-p") || !inputParser.cmdOptionExists("-d")) {
        cout << "usage: storage_service -p ${port} -d ${dataDir}" << endl;
        return -1;
    }

    google::InitGoogleLogging(argv[0]);

    const string &dataDir = inputParser.getCmdOption("-d");
    std::stringstream ss;
    ss << inputParser.getCmdOption("-p");
    ss >> port;

    RunServer(port, dataDir);

    return 0;
}