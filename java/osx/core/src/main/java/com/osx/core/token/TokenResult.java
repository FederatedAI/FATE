package com.osx.core.token;

import java.util.Map;


public class TokenResult {

    private Integer status;

    private int remaining;
    private int waitInMs;

    private long tokenId;

    private Map<String, String> attachments;

    public TokenResult() {
    }

    public TokenResult(Integer status) {
        this.status = status;
    }

    public long getTokenId() {
        return tokenId;
    }

    public void setTokenId(long tokenId) {
        this.tokenId = tokenId;
    }

    public Integer getStatus() {
        return status;
    }

    public TokenResult setStatus(Integer status) {
        this.status = status;
        return this;
    }

    public int getRemaining() {
        return remaining;
    }

    public TokenResult setRemaining(int remaining) {
        this.remaining = remaining;
        return this;
    }

    public int getWaitInMs() {
        return waitInMs;
    }

    public TokenResult setWaitInMs(int waitInMs) {
        this.waitInMs = waitInMs;
        return this;
    }

    public Map<String, String> getAttachments() {
        return attachments;
    }

    public TokenResult setAttachments(Map<String, String> attachments) {
        this.attachments = attachments;
        return this;
    }

    @Override
    public String toString() {
        return "TokenResult{" +
                "status=" + status +
                ", remaining=" + remaining +
                ", waitInMs=" + waitInMs +
                ", attachments=" + attachments +
                ", tokenId=" + tokenId +
                '}';
    }
}
