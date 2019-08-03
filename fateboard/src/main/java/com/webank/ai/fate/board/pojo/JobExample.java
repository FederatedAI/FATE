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
package com.webank.ai.fate.board.pojo;

import java.util.ArrayList;
import java.util.List;

public class JobExample {
    protected String orderByClause;

    protected boolean distinct;

    protected List<Criteria> oredCriteria;
    protected String limitClause;

    public JobExample() {
        oredCriteria = new ArrayList<Criteria>();
    }

    public String getLimitClause() {
        return limitClause;
    }

    public void setLimitClause(String limitClause) {
        this.limitClause = limitClause;
    }

    public String getOrderByClause() {
        return orderByClause;
    }

    public void setOrderByClause(String orderByClause) {
        this.orderByClause = orderByClause;
    }

    public boolean isDistinct() {
        return distinct;
    }

    public void setDistinct(boolean distinct) {
        this.distinct = distinct;
    }

    public List<Criteria> getOredCriteria() {
        return oredCriteria;
    }

    public void or(Criteria criteria) {
        oredCriteria.add(criteria);
    }

    public Criteria or() {
        Criteria criteria = createCriteriaInternal();
        oredCriteria.add(criteria);
        return criteria;
    }

    public Criteria createCriteria() {
        Criteria criteria = createCriteriaInternal();
        if (oredCriteria.size() == 0) {
            oredCriteria.add(criteria);
        }
        return criteria;
    }

    protected Criteria createCriteriaInternal() {
        Criteria criteria = new Criteria();
        return criteria;
    }

    public void clear() {
        oredCriteria.clear();
        orderByClause = null;
        distinct = false;
        limitClause = null;
    }

    protected abstract static class GeneratedCriteria {
        protected List<Criterion> criteria;

        protected GeneratedCriteria() {
            super();
            criteria = new ArrayList<Criterion>();
        }

        public boolean isValid() {
            return criteria.size() > 0;
        }

        public List<Criterion> getAllCriteria() {
            return criteria;
        }

        public List<Criterion> getCriteria() {
            return criteria;
        }

        protected void addCriterion(String condition) {
            if (condition == null) {
                throw new RuntimeException("Value for condition cannot be null");
            }
            criteria.add(new Criterion(condition));
        }

        protected void addCriterion(String condition, Object value, String property) {
            if (value == null) {
                throw new RuntimeException("Value for " + property + " cannot be null");
            }
            criteria.add(new Criterion(condition, value));
        }

        protected void addCriterion(String condition, Object value1, Object value2, String property) {
            if (value1 == null || value2 == null) {
                throw new RuntimeException("Between values for " + property + " cannot be null");
            }
            criteria.add(new Criterion(condition, value1, value2));
        }

        public Criteria andFJobIdIsNull() {
            addCriterion("f_job_id is null");
            return (Criteria) this;
        }

        public Criteria andFJobIdIsNotNull() {
            addCriterion("f_job_id is not null");
            return (Criteria) this;
        }

        public Criteria andFJobIdEqualTo(String value) {
            addCriterion("f_job_id =", value, "fJobId");
            return (Criteria) this;
        }

        public Criteria andFJobIdNotEqualTo(String value) {
            addCriterion("f_job_id <>", value, "fJobId");
            return (Criteria) this;
        }

        public Criteria andFJobIdGreaterThan(String value) {
            addCriterion("f_job_id >", value, "fJobId");
            return (Criteria) this;
        }

        public Criteria andFJobIdGreaterThanOrEqualTo(String value) {
            addCriterion("f_job_id >=", value, "fJobId");
            return (Criteria) this;
        }

        public Criteria andFJobIdLessThan(String value) {
            addCriterion("f_job_id <", value, "fJobId");
            return (Criteria) this;
        }

        public Criteria andFJobIdLessThanOrEqualTo(String value) {
            addCriterion("f_job_id <=", value, "fJobId");
            return (Criteria) this;
        }

        public Criteria andFJobIdLike(String value) {
            addCriterion("f_job_id like", value, "fJobId");
            return (Criteria) this;
        }

        public Criteria andFJobIdNotLike(String value) {
            addCriterion("f_job_id not like", value, "fJobId");
            return (Criteria) this;
        }

        public Criteria andFJobIdIn(List<String> values) {
            addCriterion("f_job_id in", values, "fJobId");
            return (Criteria) this;
        }

        public Criteria andFJobIdNotIn(List<String> values) {
            addCriterion("f_job_id not in", values, "fJobId");
            return (Criteria) this;
        }

        public Criteria andFJobIdBetween(String value1, String value2) {
            addCriterion("f_job_id between", value1, value2, "fJobId");
            return (Criteria) this;
        }

        public Criteria andFJobIdNotBetween(String value1, String value2) {
            addCriterion("f_job_id not between", value1, value2, "fJobId");
            return (Criteria) this;
        }

        public Criteria andFRoleIsNull() {
            addCriterion("f_role is null");
            return (Criteria) this;
        }

        public Criteria andFRoleIsNotNull() {
            addCriterion("f_role is not null");
            return (Criteria) this;
        }

        public Criteria andFRoleEqualTo(String value) {
            addCriterion("f_role =", value, "fRole");
            return (Criteria) this;
        }

        public Criteria andFRoleNotEqualTo(String value) {
            addCriterion("f_role <>", value, "fRole");
            return (Criteria) this;
        }

        public Criteria andFRoleGreaterThan(String value) {
            addCriterion("f_role >", value, "fRole");
            return (Criteria) this;
        }

        public Criteria andFRoleGreaterThanOrEqualTo(String value) {
            addCriterion("f_role >=", value, "fRole");
            return (Criteria) this;
        }

        public Criteria andFRoleLessThan(String value) {
            addCriterion("f_role <", value, "fRole");
            return (Criteria) this;
        }

        public Criteria andFRoleLessThanOrEqualTo(String value) {
            addCriterion("f_role <=", value, "fRole");
            return (Criteria) this;
        }

        public Criteria andFRoleLike(String value) {
            addCriterion("f_role like", value, "fRole");
            return (Criteria) this;
        }

        public Criteria andFRoleNotLike(String value) {
            addCriterion("f_role not like", value, "fRole");
            return (Criteria) this;
        }

        public Criteria andFRoleIn(List<String> values) {
            addCriterion("f_role in", values, "fRole");
            return (Criteria) this;
        }

        public Criteria andFRoleNotIn(List<String> values) {
            addCriterion("f_role not in", values, "fRole");
            return (Criteria) this;
        }

        public Criteria andFRoleBetween(String value1, String value2) {
            addCriterion("f_role between", value1, value2, "fRole");
            return (Criteria) this;
        }

        public Criteria andFRoleNotBetween(String value1, String value2) {
            addCriterion("f_role not between", value1, value2, "fRole");
            return (Criteria) this;
        }

        public Criteria andFPartyIdIsNull() {
            addCriterion("f_party_id is null");
            return (Criteria) this;
        }

        public Criteria andFPartyIdIsNotNull() {
            addCriterion("f_party_id is not null");
            return (Criteria) this;
        }

        public Criteria andFPartyIdEqualTo(String value) {
            addCriterion("f_party_id =", value, "fPartyId");
            return (Criteria) this;
        }

        public Criteria andFPartyIdNotEqualTo(String value) {
            addCriterion("f_party_id <>", value, "fPartyId");
            return (Criteria) this;
        }

        public Criteria andFPartyIdGreaterThan(String value) {
            addCriterion("f_party_id >", value, "fPartyId");
            return (Criteria) this;
        }

        public Criteria andFPartyIdGreaterThanOrEqualTo(String value) {
            addCriterion("f_party_id >=", value, "fPartyId");
            return (Criteria) this;
        }

        public Criteria andFPartyIdLessThan(String value) {
            addCriterion("f_party_id <", value, "fPartyId");
            return (Criteria) this;
        }

        public Criteria andFPartyIdLessThanOrEqualTo(String value) {
            addCriterion("f_party_id <=", value, "fPartyId");
            return (Criteria) this;
        }

        public Criteria andFPartyIdLike(String value) {
            addCriterion("f_party_id like", value, "fPartyId");
            return (Criteria) this;
        }

        public Criteria andFPartyIdNotLike(String value) {
            addCriterion("f_party_id not like", value, "fPartyId");
            return (Criteria) this;
        }

        public Criteria andFPartyIdIn(List<String> values) {
            addCriterion("f_party_id in", values, "fPartyId");
            return (Criteria) this;
        }

        public Criteria andFPartyIdNotIn(List<String> values) {
            addCriterion("f_party_id not in", values, "fPartyId");
            return (Criteria) this;
        }

        public Criteria andFPartyIdBetween(String value1, String value2) {
            addCriterion("f_party_id between", value1, value2, "fPartyId");
            return (Criteria) this;
        }

        public Criteria andFPartyIdNotBetween(String value1, String value2) {
            addCriterion("f_party_id not between", value1, value2, "fPartyId");
            return (Criteria) this;
        }

        public Criteria andFNameIsNull() {
            addCriterion("f_name is null");
            return (Criteria) this;
        }

        public Criteria andFNameIsNotNull() {
            addCriterion("f_name is not null");
            return (Criteria) this;
        }

        public Criteria andFNameEqualTo(String value) {
            addCriterion("f_name =", value, "fName");
            return (Criteria) this;
        }

        public Criteria andFNameNotEqualTo(String value) {
            addCriterion("f_name <>", value, "fName");
            return (Criteria) this;
        }

        public Criteria andFNameGreaterThan(String value) {
            addCriterion("f_name >", value, "fName");
            return (Criteria) this;
        }

        public Criteria andFNameGreaterThanOrEqualTo(String value) {
            addCriterion("f_name >=", value, "fName");
            return (Criteria) this;
        }

        public Criteria andFNameLessThan(String value) {
            addCriterion("f_name <", value, "fName");
            return (Criteria) this;
        }

        public Criteria andFNameLessThanOrEqualTo(String value) {
            addCriterion("f_name <=", value, "fName");
            return (Criteria) this;
        }

        public Criteria andFNameLike(String value) {
            addCriterion("f_name like", value, "fName");
            return (Criteria) this;
        }

        public Criteria andFNameNotLike(String value) {
            addCriterion("f_name not like", value, "fName");
            return (Criteria) this;
        }

        public Criteria andFNameIn(List<String> values) {
            addCriterion("f_name in", values, "fName");
            return (Criteria) this;
        }

        public Criteria andFNameNotIn(List<String> values) {
            addCriterion("f_name not in", values, "fName");
            return (Criteria) this;
        }

        public Criteria andFNameBetween(String value1, String value2) {
            addCriterion("f_name between", value1, value2, "fName");
            return (Criteria) this;
        }

        public Criteria andFNameNotBetween(String value1, String value2) {
            addCriterion("f_name not between", value1, value2, "fName");
            return (Criteria) this;
        }

        public Criteria andFTagIsNull() {
            addCriterion("f_tag is null");
            return (Criteria) this;
        }

        public Criteria andFTagIsNotNull() {
            addCriterion("f_tag is not null");
            return (Criteria) this;
        }

        public Criteria andFTagEqualTo(String value) {
            addCriterion("f_tag =", value, "fTag");
            return (Criteria) this;
        }

        public Criteria andFTagNotEqualTo(String value) {
            addCriterion("f_tag <>", value, "fTag");
            return (Criteria) this;
        }

        public Criteria andFTagGreaterThan(String value) {
            addCriterion("f_tag >", value, "fTag");
            return (Criteria) this;
        }

        public Criteria andFTagGreaterThanOrEqualTo(String value) {
            addCriterion("f_tag >=", value, "fTag");
            return (Criteria) this;
        }

        public Criteria andFTagLessThan(String value) {
            addCriterion("f_tag <", value, "fTag");
            return (Criteria) this;
        }

        public Criteria andFTagLessThanOrEqualTo(String value) {
            addCriterion("f_tag <=", value, "fTag");
            return (Criteria) this;
        }

        public Criteria andFTagLike(String value) {
            addCriterion("f_tag like", value, "fTag");
            return (Criteria) this;
        }

        public Criteria andFTagNotLike(String value) {
            addCriterion("f_tag not like", value, "fTag");
            return (Criteria) this;
        }

        public Criteria andFTagIn(List<String> values) {
            addCriterion("f_tag in", values, "fTag");
            return (Criteria) this;
        }

        public Criteria andFTagNotIn(List<String> values) {
            addCriterion("f_tag not in", values, "fTag");
            return (Criteria) this;
        }

        public Criteria andFTagBetween(String value1, String value2) {
            addCriterion("f_tag between", value1, value2, "fTag");
            return (Criteria) this;
        }

        public Criteria andFTagNotBetween(String value1, String value2) {
            addCriterion("f_tag not between", value1, value2, "fTag");
            return (Criteria) this;
        }

        public Criteria andFInitiatorPartyIdIsNull() {
            addCriterion("f_initiator_party_id is null");
            return (Criteria) this;
        }

        public Criteria andFInitiatorPartyIdIsNotNull() {
            addCriterion("f_initiator_party_id is not null");
            return (Criteria) this;
        }

        public Criteria andFInitiatorPartyIdEqualTo(String value) {
            addCriterion("f_initiator_party_id =", value, "fInitiatorPartyId");
            return (Criteria) this;
        }

        public Criteria andFInitiatorPartyIdNotEqualTo(String value) {
            addCriterion("f_initiator_party_id <>", value, "fInitiatorPartyId");
            return (Criteria) this;
        }

        public Criteria andFInitiatorPartyIdGreaterThan(String value) {
            addCriterion("f_initiator_party_id >", value, "fInitiatorPartyId");
            return (Criteria) this;
        }

        public Criteria andFInitiatorPartyIdGreaterThanOrEqualTo(String value) {
            addCriterion("f_initiator_party_id >=", value, "fInitiatorPartyId");
            return (Criteria) this;
        }

        public Criteria andFInitiatorPartyIdLessThan(String value) {
            addCriterion("f_initiator_party_id <", value, "fInitiatorPartyId");
            return (Criteria) this;
        }

        public Criteria andFInitiatorPartyIdLessThanOrEqualTo(String value) {
            addCriterion("f_initiator_party_id <=", value, "fInitiatorPartyId");
            return (Criteria) this;
        }

        public Criteria andFInitiatorPartyIdLike(String value) {
            addCriterion("f_initiator_party_id like", value, "fInitiatorPartyId");
            return (Criteria) this;
        }

        public Criteria andFInitiatorPartyIdNotLike(String value) {
            addCriterion("f_initiator_party_id not like", value, "fInitiatorPartyId");
            return (Criteria) this;
        }

        public Criteria andFInitiatorPartyIdIn(List<String> values) {
            addCriterion("f_initiator_party_id in", values, "fInitiatorPartyId");
            return (Criteria) this;
        }

        public Criteria andFInitiatorPartyIdNotIn(List<String> values) {
            addCriterion("f_initiator_party_id not in", values, "fInitiatorPartyId");
            return (Criteria) this;
        }

        public Criteria andFInitiatorPartyIdBetween(String value1, String value2) {
            addCriterion("f_initiator_party_id between", value1, value2, "fInitiatorPartyId");
            return (Criteria) this;
        }

        public Criteria andFInitiatorPartyIdNotBetween(String value1, String value2) {
            addCriterion("f_initiator_party_id not between", value1, value2, "fInitiatorPartyId");
            return (Criteria) this;
        }

        public Criteria andFStatusIsNull() {
            addCriterion("f_status is null");
            return (Criteria) this;
        }

        public Criteria andFStatusIsNotNull() {
            addCriterion("f_status is not null");
            return (Criteria) this;
        }

        public Criteria andFStatusEqualTo(String value) {
            addCriterion("f_status =", value, "fStatus");
            return (Criteria) this;
        }

        public Criteria andFStatusNotEqualTo(String value) {
            addCriterion("f_status <>", value, "fStatus");
            return (Criteria) this;
        }

        public Criteria andFStatusGreaterThan(String value) {
            addCriterion("f_status >", value, "fStatus");
            return (Criteria) this;
        }

        public Criteria andFStatusGreaterThanOrEqualTo(String value) {
            addCriterion("f_status >=", value, "fStatus");
            return (Criteria) this;
        }

        public Criteria andFStatusLessThan(String value) {
            addCriterion("f_status <", value, "fStatus");
            return (Criteria) this;
        }

        public Criteria andFStatusLessThanOrEqualTo(String value) {
            addCriterion("f_status <=", value, "fStatus");
            return (Criteria) this;
        }

        public Criteria andFStatusLike(String value) {
            addCriterion("f_status like", value, "fStatus");
            return (Criteria) this;
        }

        public Criteria andFStatusNotLike(String value) {
            addCriterion("f_status not like", value, "fStatus");
            return (Criteria) this;
        }

        public Criteria andFStatusIn(List<String> values) {
            addCriterion("f_status in", values, "fStatus");
            return (Criteria) this;
        }

        public Criteria andFStatusNotIn(List<String> values) {
            addCriterion("f_status not in", values, "fStatus");
            return (Criteria) this;
        }

        public Criteria andFStatusBetween(String value1, String value2) {
            addCriterion("f_status between", value1, value2, "fStatus");
            return (Criteria) this;
        }

        public Criteria andFStatusNotBetween(String value1, String value2) {
            addCriterion("f_status not between", value1, value2, "fStatus");
            return (Criteria) this;
        }

        public Criteria andFCurrentStepsIsNull() {
            addCriterion("f_current_steps is null");
            return (Criteria) this;
        }

        public Criteria andFCurrentStepsIsNotNull() {
            addCriterion("f_current_steps is not null");
            return (Criteria) this;
        }

        public Criteria andFCurrentStepsEqualTo(String value) {
            addCriterion("f_current_steps =", value, "fCurrentSteps");
            return (Criteria) this;
        }

        public Criteria andFCurrentStepsNotEqualTo(String value) {
            addCriterion("f_current_steps <>", value, "fCurrentSteps");
            return (Criteria) this;
        }

        public Criteria andFCurrentStepsGreaterThan(String value) {
            addCriterion("f_current_steps >", value, "fCurrentSteps");
            return (Criteria) this;
        }

        public Criteria andFCurrentStepsGreaterThanOrEqualTo(String value) {
            addCriterion("f_current_steps >=", value, "fCurrentSteps");
            return (Criteria) this;
        }

        public Criteria andFCurrentStepsLessThan(String value) {
            addCriterion("f_current_steps <", value, "fCurrentSteps");
            return (Criteria) this;
        }

        public Criteria andFCurrentStepsLessThanOrEqualTo(String value) {
            addCriterion("f_current_steps <=", value, "fCurrentSteps");
            return (Criteria) this;
        }

        public Criteria andFCurrentStepsLike(String value) {
            addCriterion("f_current_steps like", value, "fCurrentSteps");
            return (Criteria) this;
        }

        public Criteria andFCurrentStepsNotLike(String value) {
            addCriterion("f_current_steps not like", value, "fCurrentSteps");
            return (Criteria) this;
        }

        public Criteria andFCurrentStepsIn(List<String> values) {
            addCriterion("f_current_steps in", values, "fCurrentSteps");
            return (Criteria) this;
        }

        public Criteria andFCurrentStepsNotIn(List<String> values) {
            addCriterion("f_current_steps not in", values, "fCurrentSteps");
            return (Criteria) this;
        }

        public Criteria andFCurrentStepsBetween(String value1, String value2) {
            addCriterion("f_current_steps between", value1, value2, "fCurrentSteps");
            return (Criteria) this;
        }

        public Criteria andFCurrentStepsNotBetween(String value1, String value2) {
            addCriterion("f_current_steps not between", value1, value2, "fCurrentSteps");
            return (Criteria) this;
        }

        public Criteria andFCurrentTasksIsNull() {
            addCriterion("f_current_tasks is null");
            return (Criteria) this;
        }

        public Criteria andFCurrentTasksIsNotNull() {
            addCriterion("f_current_tasks is not null");
            return (Criteria) this;
        }

        public Criteria andFCurrentTasksEqualTo(String value) {
            addCriterion("f_current_tasks =", value, "fCurrentTasks");
            return (Criteria) this;
        }

        public Criteria andFCurrentTasksNotEqualTo(String value) {
            addCriterion("f_current_tasks <>", value, "fCurrentTasks");
            return (Criteria) this;
        }

        public Criteria andFCurrentTasksGreaterThan(String value) {
            addCriterion("f_current_tasks >", value, "fCurrentTasks");
            return (Criteria) this;
        }

        public Criteria andFCurrentTasksGreaterThanOrEqualTo(String value) {
            addCriterion("f_current_tasks >=", value, "fCurrentTasks");
            return (Criteria) this;
        }

        public Criteria andFCurrentTasksLessThan(String value) {
            addCriterion("f_current_tasks <", value, "fCurrentTasks");
            return (Criteria) this;
        }

        public Criteria andFCurrentTasksLessThanOrEqualTo(String value) {
            addCriterion("f_current_tasks <=", value, "fCurrentTasks");
            return (Criteria) this;
        }

        public Criteria andFCurrentTasksLike(String value) {
            addCriterion("f_current_tasks like", value, "fCurrentTasks");
            return (Criteria) this;
        }

        public Criteria andFCurrentTasksNotLike(String value) {
            addCriterion("f_current_tasks not like", value, "fCurrentTasks");
            return (Criteria) this;
        }

        public Criteria andFCurrentTasksIn(List<String> values) {
            addCriterion("f_current_tasks in", values, "fCurrentTasks");
            return (Criteria) this;
        }

        public Criteria andFCurrentTasksNotIn(List<String> values) {
            addCriterion("f_current_tasks not in", values, "fCurrentTasks");
            return (Criteria) this;
        }

        public Criteria andFCurrentTasksBetween(String value1, String value2) {
            addCriterion("f_current_tasks between", value1, value2, "fCurrentTasks");
            return (Criteria) this;
        }

        public Criteria andFCurrentTasksNotBetween(String value1, String value2) {
            addCriterion("f_current_tasks not between", value1, value2, "fCurrentTasks");
            return (Criteria) this;
        }

        public Criteria andFProgressIsNull() {
            addCriterion("f_progress is null");
            return (Criteria) this;
        }

        public Criteria andFProgressIsNotNull() {
            addCriterion("f_progress is not null");
            return (Criteria) this;
        }

        public Criteria andFProgressEqualTo(Integer value) {
            addCriterion("f_progress =", value, "fProgress");
            return (Criteria) this;
        }

        public Criteria andFProgressNotEqualTo(Integer value) {
            addCriterion("f_progress <>", value, "fProgress");
            return (Criteria) this;
        }

        public Criteria andFProgressGreaterThan(Integer value) {
            addCriterion("f_progress >", value, "fProgress");
            return (Criteria) this;
        }

        public Criteria andFProgressGreaterThanOrEqualTo(Integer value) {
            addCriterion("f_progress >=", value, "fProgress");
            return (Criteria) this;
        }

        public Criteria andFProgressLessThan(Integer value) {
            addCriterion("f_progress <", value, "fProgress");
            return (Criteria) this;
        }

        public Criteria andFProgressLessThanOrEqualTo(Integer value) {
            addCriterion("f_progress <=", value, "fProgress");
            return (Criteria) this;
        }

        public Criteria andFProgressIn(List<Integer> values) {
            addCriterion("f_progress in", values, "fProgress");
            return (Criteria) this;
        }

        public Criteria andFProgressNotIn(List<Integer> values) {
            addCriterion("f_progress not in", values, "fProgress");
            return (Criteria) this;
        }

        public Criteria andFProgressBetween(Integer value1, Integer value2) {
            addCriterion("f_progress between", value1, value2, "fProgress");
            return (Criteria) this;
        }

        public Criteria andFProgressNotBetween(Integer value1, Integer value2) {
            addCriterion("f_progress not between", value1, value2, "fProgress");
            return (Criteria) this;
        }

        public Criteria andFCreateTimeIsNull() {
            addCriterion("f_create_time is null");
            return (Criteria) this;
        }

        public Criteria andFCreateTimeIsNotNull() {
            addCriterion("f_create_time is not null");
            return (Criteria) this;
        }

        public Criteria andFCreateTimeEqualTo(Long value) {
            addCriterion("f_create_time =", value, "fCreateTime");
            return (Criteria) this;
        }

        public Criteria andFCreateTimeNotEqualTo(Long value) {
            addCriterion("f_create_time <>", value, "fCreateTime");
            return (Criteria) this;
        }

        public Criteria andFCreateTimeGreaterThan(Long value) {
            addCriterion("f_create_time >", value, "fCreateTime");
            return (Criteria) this;
        }

        public Criteria andFCreateTimeGreaterThanOrEqualTo(Long value) {
            addCriterion("f_create_time >=", value, "fCreateTime");
            return (Criteria) this;
        }

        public Criteria andFCreateTimeLessThan(Long value) {
            addCriterion("f_create_time <", value, "fCreateTime");
            return (Criteria) this;
        }

        public Criteria andFCreateTimeLessThanOrEqualTo(Long value) {
            addCriterion("f_create_time <=", value, "fCreateTime");
            return (Criteria) this;
        }

        public Criteria andFCreateTimeIn(List<Long> values) {
            addCriterion("f_create_time in", values, "fCreateTime");
            return (Criteria) this;
        }

        public Criteria andFCreateTimeNotIn(List<Long> values) {
            addCriterion("f_create_time not in", values, "fCreateTime");
            return (Criteria) this;
        }

        public Criteria andFCreateTimeBetween(Long value1, Long value2) {
            addCriterion("f_create_time between", value1, value2, "fCreateTime");
            return (Criteria) this;
        }

        public Criteria andFCreateTimeNotBetween(Long value1, Long value2) {
            addCriterion("f_create_time not between", value1, value2, "fCreateTime");
            return (Criteria) this;
        }

        public Criteria andFUpdateTimeIsNull() {
            addCriterion("f_update_time is null");
            return (Criteria) this;
        }

        public Criteria andFUpdateTimeIsNotNull() {
            addCriterion("f_update_time is not null");
            return (Criteria) this;
        }

        public Criteria andFUpdateTimeEqualTo(Long value) {
            addCriterion("f_update_time =", value, "fUpdateTime");
            return (Criteria) this;
        }

        public Criteria andFUpdateTimeNotEqualTo(Long value) {
            addCriterion("f_update_time <>", value, "fUpdateTime");
            return (Criteria) this;
        }

        public Criteria andFUpdateTimeGreaterThan(Long value) {
            addCriterion("f_update_time >", value, "fUpdateTime");
            return (Criteria) this;
        }

        public Criteria andFUpdateTimeGreaterThanOrEqualTo(Long value) {
            addCriterion("f_update_time >=", value, "fUpdateTime");
            return (Criteria) this;
        }

        public Criteria andFUpdateTimeLessThan(Long value) {
            addCriterion("f_update_time <", value, "fUpdateTime");
            return (Criteria) this;
        }

        public Criteria andFUpdateTimeLessThanOrEqualTo(Long value) {
            addCriterion("f_update_time <=", value, "fUpdateTime");
            return (Criteria) this;
        }

        public Criteria andFUpdateTimeIn(List<Long> values) {
            addCriterion("f_update_time in", values, "fUpdateTime");
            return (Criteria) this;
        }

        public Criteria andFUpdateTimeNotIn(List<Long> values) {
            addCriterion("f_update_time not in", values, "fUpdateTime");
            return (Criteria) this;
        }

        public Criteria andFUpdateTimeBetween(Long value1, Long value2) {
            addCriterion("f_update_time between", value1, value2, "fUpdateTime");
            return (Criteria) this;
        }

        public Criteria andFUpdateTimeNotBetween(Long value1, Long value2) {
            addCriterion("f_update_time not between", value1, value2, "fUpdateTime");
            return (Criteria) this;
        }

        public Criteria andFStartTimeIsNull() {
            addCriterion("f_start_time is null");
            return (Criteria) this;
        }

        public Criteria andFStartTimeIsNotNull() {
            addCriterion("f_start_time is not null");
            return (Criteria) this;
        }

        public Criteria andFStartTimeEqualTo(Long value) {
            addCriterion("f_start_time =", value, "fStartTime");
            return (Criteria) this;
        }

        public Criteria andFStartTimeNotEqualTo(Long value) {
            addCriterion("f_start_time <>", value, "fStartTime");
            return (Criteria) this;
        }

        public Criteria andFStartTimeGreaterThan(Long value) {
            addCriterion("f_start_time >", value, "fStartTime");
            return (Criteria) this;
        }

        public Criteria andFStartTimeGreaterThanOrEqualTo(Long value) {
            addCriterion("f_start_time >=", value, "fStartTime");
            return (Criteria) this;
        }

        public Criteria andFStartTimeLessThan(Long value) {
            addCriterion("f_start_time <", value, "fStartTime");
            return (Criteria) this;
        }

        public Criteria andFStartTimeLessThanOrEqualTo(Long value) {
            addCriterion("f_start_time <=", value, "fStartTime");
            return (Criteria) this;
        }

        public Criteria andFStartTimeIn(List<Long> values) {
            addCriterion("f_start_time in", values, "fStartTime");
            return (Criteria) this;
        }

        public Criteria andFStartTimeNotIn(List<Long> values) {
            addCriterion("f_start_time not in", values, "fStartTime");
            return (Criteria) this;
        }

        public Criteria andFStartTimeBetween(Long value1, Long value2) {
            addCriterion("f_start_time between", value1, value2, "fStartTime");
            return (Criteria) this;
        }

        public Criteria andFStartTimeNotBetween(Long value1, Long value2) {
            addCriterion("f_start_time not between", value1, value2, "fStartTime");
            return (Criteria) this;
        }

        public Criteria andFEndTimeIsNull() {
            addCriterion("f_end_time is null");
            return (Criteria) this;
        }

        public Criteria andFEndTimeIsNotNull() {
            addCriterion("f_end_time is not null");
            return (Criteria) this;
        }

        public Criteria andFEndTimeEqualTo(Long value) {
            addCriterion("f_end_time =", value, "fEndTime");
            return (Criteria) this;
        }

        public Criteria andFEndTimeNotEqualTo(Long value) {
            addCriterion("f_end_time <>", value, "fEndTime");
            return (Criteria) this;
        }

        public Criteria andFEndTimeGreaterThan(Long value) {
            addCriterion("f_end_time >", value, "fEndTime");
            return (Criteria) this;
        }

        public Criteria andFEndTimeGreaterThanOrEqualTo(Long value) {
            addCriterion("f_end_time >=", value, "fEndTime");
            return (Criteria) this;
        }

        public Criteria andFEndTimeLessThan(Long value) {
            addCriterion("f_end_time <", value, "fEndTime");
            return (Criteria) this;
        }

        public Criteria andFEndTimeLessThanOrEqualTo(Long value) {
            addCriterion("f_end_time <=", value, "fEndTime");
            return (Criteria) this;
        }

        public Criteria andFEndTimeIn(List<Long> values) {
            addCriterion("f_end_time in", values, "fEndTime");
            return (Criteria) this;
        }

        public Criteria andFEndTimeNotIn(List<Long> values) {
            addCriterion("f_end_time not in", values, "fEndTime");
            return (Criteria) this;
        }

        public Criteria andFEndTimeBetween(Long value1, Long value2) {
            addCriterion("f_end_time between", value1, value2, "fEndTime");
            return (Criteria) this;
        }

        public Criteria andFEndTimeNotBetween(Long value1, Long value2) {
            addCriterion("f_end_time not between", value1, value2, "fEndTime");
            return (Criteria) this;
        }

        public Criteria andFElapsedIsNull() {
            addCriterion("f_elapsed is null");
            return (Criteria) this;
        }

        public Criteria andFElapsedIsNotNull() {
            addCriterion("f_elapsed is not null");
            return (Criteria) this;
        }

        public Criteria andFElapsedEqualTo(Long value) {
            addCriterion("f_elapsed =", value, "fElapsed");
            return (Criteria) this;
        }

        public Criteria andFElapsedNotEqualTo(Long value) {
            addCriterion("f_elapsed <>", value, "fElapsed");
            return (Criteria) this;
        }

        public Criteria andFElapsedGreaterThan(Long value) {
            addCriterion("f_elapsed >", value, "fElapsed");
            return (Criteria) this;
        }

        public Criteria andFElapsedGreaterThanOrEqualTo(Long value) {
            addCriterion("f_elapsed >=", value, "fElapsed");
            return (Criteria) this;
        }

        public Criteria andFElapsedLessThan(Long value) {
            addCriterion("f_elapsed <", value, "fElapsed");
            return (Criteria) this;
        }

        public Criteria andFElapsedLessThanOrEqualTo(Long value) {
            addCriterion("f_elapsed <=", value, "fElapsed");
            return (Criteria) this;
        }

        public Criteria andFElapsedIn(List<Long> values) {
            addCriterion("f_elapsed in", values, "fElapsed");
            return (Criteria) this;
        }

        public Criteria andFElapsedNotIn(List<Long> values) {
            addCriterion("f_elapsed not in", values, "fElapsed");
            return (Criteria) this;
        }

        public Criteria andFElapsedBetween(Long value1, Long value2) {
            addCriterion("f_elapsed between", value1, value2, "fElapsed");
            return (Criteria) this;
        }

        public Criteria andFElapsedNotBetween(Long value1, Long value2) {
            addCriterion("f_elapsed not between", value1, value2, "fElapsed");
            return (Criteria) this;
        }

        public Criteria andFRunIpIsNull() {
            addCriterion("f_run_ip is null");
            return (Criteria) this;
        }

        public Criteria andFRunIpIsNotNull() {
            addCriterion("f_run_ip is not null");
            return (Criteria) this;
        }

        public Criteria andFRunIpEqualTo(String value) {
            addCriterion("f_run_ip =", value, "fRunIp");
            return (Criteria) this;
        }

        public Criteria andFRunIpNotEqualTo(String value) {
            addCriterion("f_run_ip <>", value, "fRunIp");
            return (Criteria) this;
        }

        public Criteria andFRunIpGreaterThan(String value) {
            addCriterion("f_run_ip >", value, "fRunIp");
            return (Criteria) this;
        }

        public Criteria andFRunIpGreaterThanOrEqualTo(String value) {
            addCriterion("f_run_ip >=", value, "fRunIp");
            return (Criteria) this;
        }

        public Criteria andFRunIpLessThan(String value) {
            addCriterion("f_run_ip <", value, "fRunIp");
            return (Criteria) this;
        }

        public Criteria andFRunIpLessThanOrEqualTo(String value) {
            addCriterion("f_run_ip <=", value, "fRunIp");
            return (Criteria) this;
        }

        public Criteria andFRunIpLike(String value) {
            addCriterion("f_run_ip like", value, "fRunIp");
            return (Criteria) this;
        }

        public Criteria andFRunIpNotLike(String value) {
            addCriterion("f_run_ip not like", value, "fRunIp");
            return (Criteria) this;
        }

        public Criteria andFRunIpIn(List<String> values) {
            addCriterion("f_run_ip in", values, "fRunIp");
            return (Criteria) this;
        }

        public Criteria andFRunIpNotIn(List<String> values) {
            addCriterion("f_run_ip not in", values, "fRunIp");
            return (Criteria) this;
        }

        public Criteria andFRunIpBetween(String value1, String value2) {
            addCriterion("f_run_ip between", value1, value2, "fRunIp");
            return (Criteria) this;
        }

        public Criteria andFRunIpNotBetween(String value1, String value2) {
            addCriterion("f_run_ip not between", value1, value2, "fRunIp");
            return (Criteria) this;
        }
    }

    public static class Criteria extends GeneratedCriteria {

        protected Criteria() {
            super();
        }
    }

    public static class Criterion {
        private String condition;

        private Object value;

        private Object secondValue;

        private boolean noValue;

        private boolean singleValue;

        private boolean betweenValue;

        private boolean listValue;

        private String typeHandler;

        protected Criterion(String condition) {
            super();
            this.condition = condition;
            this.typeHandler = null;
            this.noValue = true;
        }

        protected Criterion(String condition, Object value, String typeHandler) {
            super();
            this.condition = condition;
            this.value = value;
            this.typeHandler = typeHandler;
            if (value instanceof List<?>) {
                this.listValue = true;
            } else {
                this.singleValue = true;
            }
        }

        protected Criterion(String condition, Object value) {
            this(condition, value, null);
        }

        protected Criterion(String condition, Object value, Object secondValue, String typeHandler) {
            super();
            this.condition = condition;
            this.value = value;
            this.secondValue = secondValue;
            this.typeHandler = typeHandler;
            this.betweenValue = true;
        }

        protected Criterion(String condition, Object value, Object secondValue) {
            this(condition, value, secondValue, null);
        }

        public String getCondition() {
            return condition;
        }

        public Object getValue() {
            return value;
        }

        public Object getSecondValue() {
            return secondValue;
        }

        public boolean isNoValue() {
            return noValue;
        }

        public boolean isSingleValue() {
            return singleValue;
        }

        public boolean isBetweenValue() {
            return betweenValue;
        }

        public boolean isListValue() {
            return listValue;
        }

        public String getTypeHandler() {
            return typeHandler;
        }
    }
}