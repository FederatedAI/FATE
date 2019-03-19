package com.webank.ai.fate.serving;

import com.webank.ai.fate.serving.manger.ModelManager;
import com.webank.ai.fate.core.network.grpc.client.ClientPool;
import com.webank.ai.fate.serving.service.ModelService;
import com.webank.ai.fate.serving.service.PredictService;
import com.webank.ai.fate.core.utils.Configuration;
import com.webank.ai.fate.serving.service.ProxyService;
import io.grpc.Server;
import io.grpc.ServerBuilder;
import java.io.IOException;
import java.util.ArrayList;

import org.apache.commons.cli.*;
import org.apache.commons.lang3.StringUtils;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

public class ServingServer {
    private static final Logger LOGGER = LogManager.getLogger();
    private Server server;
    private String confPath;

    public ServingServer(String confPath){
        this.confPath = confPath;
    }

    private void start() throws IOException {
        if (StringUtils.isEmpty(this.confPath)){
            // just a test
            this.confPath = String.format("%s/serving.properties", this.getClass().getResource("/").getPath());
        }
        new Configuration(this.confPath).load();
        ModelManager modelManager = new ModelManager();
        //modelManager.updatePool();

        int port = Integer.parseInt(Configuration.getProperty("port"));
        server = ServerBuilder.forPort(port)
                .addService(new PredictService())
                .addService(new ModelService())
                .addService(new ProxyService())
                .build();
        LOGGER.info("Server started listening on port: {}, use configuration: {}", port, this.confPath);
        this.initClientPool();
        server.start();
        Runtime.getRuntime().addShutdownHook(new Thread() {
            @Override
            public void run() {
                LOGGER.info("*** shutting down gRPC server since JVM is shutting down");
                ServingServer.this.stop();
                LOGGER.info("*** server shut down");
            }
        });
    }

    private void stop() {
        if (server != null) {
            server.shutdown();
        }
    }

    private void blockUntilShutdown() throws InterruptedException {
        if (server != null) {
            server.awaitTermination();
        }
    }

    private void initClientPool(){
        ArrayList<String> allAddress = new ArrayList<>();
        allAddress.add(Configuration.getProperty("proxy"));
        allAddress.add(Configuration.getProperty("serving"));
        new Thread(new Runnable() {
            @Override
            public void run() {
                ClientPool.init_pool(allAddress);
            }
        }).start();
        LOGGER.info("Finish init client pool");
    }

    public static void main(String[] args){
        try{
            Options options = new Options();
            Option option = Option.builder("c")
                    .longOpt("config")
                    .argName("file")
                    .hasArg()
                    .numberOfArgs(1)
                    .desc("configuration file")
                    .build();
            options.addOption(option);
            CommandLineParser parser = new DefaultParser();
            CommandLine cmd = parser.parse(options, args);

            ServingServer a = new ServingServer(cmd.getOptionValue("c"));
            a.start();
            a.blockUntilShutdown();
        }
        catch (Exception ex){
            ex.printStackTrace();
        }
    }
}
