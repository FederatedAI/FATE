package com.webank.ai.fate.board.utils;

import java.io.*;

public class ReadJson {
    private ReadJson() {
    }

    public static String readJsonFile(String filename) {

        StringBuilder strFile = new StringBuilder();
        File jsonFile = new File(filename);


        FileInputStream fileInputStream = null;
        InputStreamReader inputStreamReader = null;
        BufferedReader bufferedReader = null;
        try {
            fileInputStream = new FileInputStream(jsonFile);
            inputStreamReader = new InputStreamReader(fileInputStream, "GBK");
            bufferedReader = new BufferedReader(inputStreamReader);
            String str;
            while ((str = bufferedReader.readLine()) != null) {
                strFile.append(str);
            }
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            if (bufferedReader != null) {
                try {
                    bufferedReader.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
        return strFile.toString();
    }
}
