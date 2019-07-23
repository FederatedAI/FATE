package com.webank.ai.fate.board.conf;

import javax.websocket.HandshakeResponse;
import javax.websocket.server.HandshakeRequest;
import javax.websocket.server.ServerEndpointConfig;
import java.util.Collections;

public class Configurator extends ServerEndpointConfig.Configurator {


    @Override
    public void modifyHandshake(ServerEndpointConfig sec, HandshakeRequest request,
                                HandshakeResponse response) {

        response.getHeaders().put("Access-Control-Allow-Origin", Collections.singletonList("*"));
        response.getHeaders().put("Access-Control-Allow-Methods", Collections.singletonList("POST, GET, OPTIONS, DELETE"));
        response.getHeaders().put("Access-Control-Max-Age", Collections.singletonList("3600"));
        response.getHeaders().put("Access-Control-Allow-Headers", Collections.singletonList("x-requested-with"));


    }
}