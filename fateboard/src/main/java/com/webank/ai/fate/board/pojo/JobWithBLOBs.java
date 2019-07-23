package com.webank.ai.fate.board.pojo;

public class JobWithBLOBs extends Job {
    private String fDescription;

    private String fRoles;

    private String fDsl;

    private String fRuntimeConf;

    public String getfDescription() {
        return fDescription;
    }

    public void setfDescription(String fDescription) {
        this.fDescription = fDescription == null ? null : fDescription.trim();
    }

    public String getfRoles() {
        return fRoles;
    }

    public void setfRoles(String fRoles) {
        this.fRoles = fRoles == null ? null : fRoles.trim();
    }

    public String getfDsl() {
        return fDsl;
    }

    public void setfDsl(String fDsl) {
        this.fDsl = fDsl == null ? null : fDsl.trim();
    }

    public String getfRuntimeConf() {
        return fRuntimeConf;
    }

    public void setfRuntimeConf(String fRuntimeConf) {
        this.fRuntimeConf = fRuntimeConf == null ? null : fRuntimeConf.trim();
    }
}