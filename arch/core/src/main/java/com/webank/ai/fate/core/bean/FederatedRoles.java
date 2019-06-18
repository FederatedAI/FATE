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

package com.webank.ai.fate.core.bean;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class FederatedRoles {
    private Map<String, List<String>> role;

    public FederatedRoles() {
        this.role = new HashMap<>();
    }

    public Map<String, List<String>> getAllRole() {
        return role;
    }

    public List<String> getRole(String role) {
        return this.role.get(role);
    }

    public void setRole(String role, List<String> partyIds) {
        this.role.put(role, partyIds);
    }

    public void addParty(String role, String partyId) {
        this.role.get(role).add(partyId);
    }
}
