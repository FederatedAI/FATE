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



