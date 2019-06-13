#
#  Copyright 2019 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

HOST_SYSTEM = $(shell uname | cut -f 1 -d_)
SYSTEM ?= $(HOST_SYSTEM)
CXX = g++
INCLUDES += -I. -I./third_party/include
CPPFLAGS += -lprotobuf -lgrpc -lgrpc++ $(INCLUDES) $(LDFLAGS)
CXXFLAGS += -std=c++11

ifeq ($(SYSTEM),Darwin)
LDFLAGS += -L/usr/local/lib -lprotobuf -lgrpc++ -lgrpc\
           -lgrpc++_reflection\
           -ldl
else
LDFLAGS += -L/usr/local/lib -lprotobuf -lgrpc++ -lgrpc\
           -Wl,--no-as-needed -lgrpc++_reflection -Wl,--as-needed\
           -ldl
endif

LDFLAGS += -L./third_party/lib -llmdb -lboost_system -lboost_filesystem -lglog -lgpr

PROTOC = `which protoc`
GRPC_CPP_PLUGIN = `which grpc_cpp_plugin`
GRPC_CPP_PLUGIN_PATH ?= `which $(GRPC_CPP_PLUGIN)`

GRPC_GEN_OUTPATH = gen
PROTOS_PATH = proto
BIN = bin

TARGET = storage-service

DIR = ./src ./gen ./

OBJPATH = ./OBJPATH

FILES=$(foreach dir,$(DIR),$(wildcard $(dir)/*.c))

SRCS = $(wildcard *.cc)
OBJS = $(patsubst %.c,%.o,$(FILES))
VERSION = 0.3

vpath %.proto $(PROTOS_PATH)
vpath %.cc ./src
vpath %.cc ./gen
vpath %.cc ./third_party/lmdb-safe
vpath %.cc .

all: system-check $(OBJS) $(TARGET)

$(TARGET): lmdb-safe.o storage.pb.o storage.grpc.pb.o ExceptionHandler.o StoreInfo.o LMDBStore.o LMDBServicer.o $(TARGET).o
	-mv *.pb.cc src
	$(CXX) -o $@-$(VERSION) $^ $(LDFLAGS) $(INCLUDES)
	-ln -sf $@-$(VERSION) $@

.PRECIOUS: %.grpc.pb.cc
%.grpc.pb.cc: %.proto
	$(PROTOC) -I $(PROTOS_PATH) --grpc_out=. --plugin=protoc-gen-grpc=$(GRPC_CPP_PLUGIN) $<

.PRECIOUS: %.pb.cc
%.pb.cc: %.proto
	$(PROTOC) -I $(PROTOS_PATH) --cpp_out=. $<

.PRECIOUS: %.cc
	$(CXX) $^ $(LDFLAGS) -o $(BIN)/$@

.PHONY: clean
clean:
	-rm *.o *.pb.cc *.pb.h $(TARGET)-$(VERSION) $(TARGET)
	-rm -r gen
	-rm src/*.pb.cc src/*.pb.h


# The following is to test your system and ensure a smoother experience.
# They are by no means necessary to actually compile a grpc-enabled software.

PROTOC_CMD = which $(PROTOC)
PROTOC_CHECK_CMD = $(PROTOC) --version | grep -q libprotoc.3
PLUGIN_CHECK_CMD = which $(GRPC_CPP_PLUGIN)
HAS_PROTOC = $(shell $(PROTOC_CMD) > /dev/null && echo true || echo false)
ifeq ($(HAS_PROTOC),true)
HAS_VALID_PROTOC = $(shell $(PROTOC_CHECK_CMD) 2> /dev/null && echo true || echo false)
endif
HAS_PLUGIN = $(shell $(PLUGIN_CHECK_CMD) > /dev/null && echo true || echo false)

SYSTEM_OK = false
ifeq ($(HAS_VALID_PROTOC),true)
ifeq ($(HAS_PLUGIN),true)
SYSTEM_OK = true
endif
endif

system-check:
ifneq ($(HAS_VALID_PROTOC),true)
	@echo " DEPENDENCY ERROR"
	@echo
	@echo "You don't have protoc 3.0.0 installed in your path."
	@echo "Please install Google protocol buffers 3.0.0 and its compiler."
	@echo "You can find it here:"
	@echo
	@echo "   https://github.com/google/protobuf/releases/tag/v3.0.0"
	@echo
	@echo "Here is what I get when trying to evaluate your version of protoc:"
	@echo
	-$(PROTOC) --version
	@echo
	@echo
endif
ifneq ($(HAS_PLUGIN),true)
	@echo " DEPENDENCY ERROR"
	@echo
	@echo "You don't have the grpc c++ protobuf plugin installed in your path."
	@echo "Please install grpc. You can find it here:"
	@echo
	@echo "   https://github.com/grpc/grpc"
	@echo
	@echo "Here is what I get when trying to detect if you have the plugin:"
	@echo
	-which $(GRPC_CPP_PLUGIN)
	@echo
	@echo
endif
ifneq ($(SYSTEM_OK),true)
	@false
endif
	mkdir -p $(GRPC_GEN_OUTPATH)
