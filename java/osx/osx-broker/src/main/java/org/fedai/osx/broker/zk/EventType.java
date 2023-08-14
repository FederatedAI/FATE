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

package org.fedai.osx.broker.zk;

import org.apache.zookeeper.Watcher;


public enum EventType {
    None(-1),
    NodeCreated(1),
    NodeDeleted(2),
    NodeDataChanged(3),
    NodeChildrenChanged(4),
    CONNECTION_SUSPENDED(11),
    CONNECTION_RECONNECTED(12),
    CONNECTION_LOST(12),
    INITIALIZED(10);

    /**
     * Integer representation of value
     */
    private final int intValue;
    // for sending over wire

    EventType(int intValue) {
        this.intValue = intValue;
    }

    public static Watcher.Event.EventType fromInt(int intValue) {
        switch (intValue) {
            case -1:
                return Watcher.Event.EventType.None;
            case 1:
                return Watcher.Event.EventType.NodeCreated;
            case 2:
                return Watcher.Event.EventType.NodeDeleted;
            case 3:
                return Watcher.Event.EventType.NodeDataChanged;
            case 4:
                return Watcher.Event.EventType.NodeChildrenChanged;

            default:
                throw new RuntimeException("Invalid integer value for conversion to EventType");
        }
    }

    public int getIntValue() {
        return intValue;
    }
}
