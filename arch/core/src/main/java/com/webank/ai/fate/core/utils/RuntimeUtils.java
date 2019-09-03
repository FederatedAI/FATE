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

package com.webank.ai.fate.core.utils;

import com.webank.ai.fate.core.server.DefaultServerConf;
import com.webank.ai.fate.core.server.ServerConf;
import org.apache.commons.lang3.StringUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;
import java.io.IOException;
import java.net.*;
import java.util.Collections;
import java.util.Enumeration;

@Component
public class RuntimeUtils {
    @Autowired
    private ServerConf serverConf;

    private static String myIpAndPort = null;
    private static String siteLocalAddress = null;

    @PostConstruct
    private void init() {
        if (serverConf == null) {
            serverConf = new DefaultServerConf();
        }
    }


    /**
     * validation of InetAddress
     * @param nif Network Interface
     * @param adr Internet Protocol (IP) address
     * @return valid or in valid
     * @throws SocketException
     */
    private boolean checkAddrValid(NetworkInterface nif, InetAddress adr)
        throws SocketException {
      return adr != null && !adr.isLoopbackAddress() && (nif.isPointToPoint() || !adr.isLinkLocalAddress());
    }

    /**
     * get valid local ip address
     * @return Internet Protocol (IP) address
     * @throws SocketException
     * @throws UnknownHostException
     */
    private InetAddress getLocalAdr() throws SocketException, UnknownHostException {
      String os = System.getProperty("os.name").toLowerCase();
      if (os.contains("nix") || os.contains("nux")) {
        Enumeration<NetworkInterface> nifs = null;
        nifs = NetworkInterface.getNetworkInterfaces();

        if (null == nifs) {
          return null;
        }
        while (nifs.hasMoreElements()) {
          NetworkInterface nif = nifs.nextElement();
          Enumeration<InetAddress> adrs = nif.getInetAddresses();

          while (adrs.hasMoreElements()) {
            InetAddress adr = adrs.nextElement();
            if (checkAddrValid(nif, adr)) {
              return adr;
            }
          }
        }
      } else {
        return InetAddress.getLocalHost();
      }
      return null;
    }


    public String getMySiteLocalAddress() {
      if (siteLocalAddress == null) {
        try {
          InetAddress inetAddress = getLocalAdr();
          siteLocalAddress = null == inetAddress ? "127.0.0.1" : inetAddress.getHostAddress();
        } catch (SocketException | UnknownHostException e) {
          siteLocalAddress = "127.0.0.1";
        }
      }
      return siteLocalAddress;
    }

    public String getMySiteLocalIpAndPort() {
        if (myIpAndPort == null) {
            myIpAndPort = getMySiteLocalAddress() + ":" + serverConf.getPort();
        }

        return myIpAndPort;
    }

    public boolean isPortAvailable(int port) {
        try (ServerSocket ignored = new ServerSocket(port)) {
            return true;
        } catch (IOException ignored) {
            return false;
        }
    }
}
