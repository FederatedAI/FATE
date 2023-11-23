package org.fedai.osx.core.service;


public interface ApplicationStartedRunner {

    default int getRunnerSequenceId(){
        return 0;
    }

    void run(String[] args) throws Exception;
}
