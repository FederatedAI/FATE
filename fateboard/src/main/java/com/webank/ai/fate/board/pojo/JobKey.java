package com.webank.ai.fate.board.pojo;

public class JobKey {
    private String fJobId;

    private String fRole;

    private String fPartyId;

    public String getfJobId() {
        return fJobId;
    }

    public void setfJobId(String fJobId) {
        this.fJobId = fJobId == null ? null : fJobId.trim();
    }

    public String getfRole() {
        return fRole;
    }

    public void setfRole(String fRole) {
        this.fRole = fRole == null ? null : fRole.trim();
    }

    public String getfPartyId() {
        return fPartyId;
    }

    public void setfPartyId(String fPartyId) {
        this.fPartyId = fPartyId == null ? null : fPartyId.trim();
    }
}