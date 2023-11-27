package org.fedai.osx.broker.util;

import org.apache.commons.net.telnet.TelnetClient;

public class TelnetUtil {

    public static boolean tryTelnet(String host, int port) {
        TelnetClient telnetClient = new TelnetClient("vt200");
        telnetClient.setDefaultTimeout(5000);
        boolean isConnected = false;
        try {
            telnetClient.connect(host, port);
            isConnected = true;
            telnetClient.disconnect();
        } catch (Exception e) {
            //e.printStackTrace();
        }
        return isConnected;
    }

}
