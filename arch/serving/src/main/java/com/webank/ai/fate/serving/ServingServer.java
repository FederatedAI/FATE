package com.webank.ai.fate.serving;

import com.webank.ai.fate.serving.service.PredictService;
import io.grpc.Server;
import io.grpc.ServerBuilder;
import java.io.IOException;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

public class ServingServer {
    private static final Logger LOGGER = LogManager.getLogger();
    private Server server;

    private void start() throws IOException {
        int port = 50051;
        server = ServerBuilder.forPort(port)
                .addService(new PredictService())
                .build();
        LOGGER.info("Server started listening on port: 50051, data dir: xxx");
        server.start();
        Runtime.getRuntime().addShutdownHook(new Thread() {
            @Override
            public void run() {
                System.err.println("*** shutting down gRPC server since JVM is shutting down");
                ServingServer.this.stop();
                System.err.println("*** server shut down");
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

    public static void main(String[] args){
        try{
            ServingServer a = new ServingServer();
            a.start();
            a.blockUntilShutdown();
        }
        catch (Exception ex){
            ex.printStackTrace();
        }
    }
}
