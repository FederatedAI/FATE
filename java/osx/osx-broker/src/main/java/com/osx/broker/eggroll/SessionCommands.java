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
package com.osx.broker.eggroll;

public class SessionCommands {
    static String prefix = "v1/cluster-manager/session";
    static CommandURI getOrCreateSession = new CommandURI(prefix, "getOrCreateSession");
    static CommandURI getSession = new CommandURI(prefix, "getSession");
    static CommandURI registerSession = new CommandURI(prefix, "registerSession");
    static CommandURI getSessionServerNodes = new CommandURI(prefix, "getSessionServerNodes");
    static CommandURI getSessionRolls = new CommandURI(prefix, "getSessionRolls");
    static CommandURI getSessionEggs = new CommandURI(prefix, "getSessionEggs");
    static CommandURI heartbeat = new CommandURI(prefix, "heartbeat");
    static CommandURI stopSession = new CommandURI(prefix, "stopSession");
    static CommandURI killSession = new CommandURI(prefix, "killSession");
    static CommandURI killAllSessions = new CommandURI(prefix, "killAllSessions");

}



