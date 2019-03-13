package com.webank.ai.fate.serving;

public abstract class BaseServer<K, V> {
    public V mytest;
    public void test(V input){
        this.mytest = input;
        System.out.println("this is base server");
    }
}
