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
package com.webank.ai.fate.board.utils;

import java.io.Serializable;
import java.util.List;

public class PageBean<T> implements Serializable {

    private long pageNum;
    private long pageSize;
    private long totalRecord;


    private long totalPage;
    private long startIndex;


    private List<T> list;


    private long start;
    private long end;


    public PageBean(long pageNum, long pageSize, long totalRecord) {
        this.pageNum = pageNum;
        this.pageSize = pageSize;
        this.totalRecord = totalRecord;


        if (totalRecord % pageSize == 0) {

            this.totalPage = totalRecord / pageSize;
        } else {

            this.totalPage = totalRecord / pageSize + 1;
        }

        this.startIndex = (pageNum - 1) * pageSize;

        this.start = 1;
        this.end = 5;


        if (totalPage <= 5) {

            this.end = this.totalPage;
        } else {

            this.start = pageNum - 2;
            this.end = pageNum + 2;

            if (start < 0) {

                this.start = 1;
                this.end = 5;
            }
            if (end > this.totalPage) {

                this.end = totalPage;
                this.start = end - 5;
            }
        }
    }


    public long getPageNum() {
        return pageNum;
    }

    public void setPageNum(long pageNum) {
        this.pageNum = pageNum;
    }

    public long getPageSize() {
        return pageSize;
    }

    public void setPageSize(long pageSize) {
        this.pageSize = pageSize;
    }

    public long getTotalRecord() {
        return totalRecord;
    }

    public void setTotalRecord(long totalRecord) {
        this.totalRecord = totalRecord;
    }

    public long getTotalPage() {
        return totalPage;
    }

    public void setTotalPage(long totalPage) {
        this.totalPage = totalPage;
    }

    public long getStartIndex() {
        return startIndex;
    }

    public void setStartIndex(long startIndex) {
        this.startIndex = startIndex;
    }

    public List<T> getList() {
        return list;
    }

    public void setList(List<T> list) {
        this.list = list;
    }

    public long getStart() {
        return start;
    }

    public void setStart(long start) {
        this.start = start;
    }

    public long getEnd() {
        return end;
    }

    public void setEnd(long end) {
        this.end = end;
    }

    @Override
    public String toString() {
        return "PageBean{" +
                "pageNum=" + pageNum +
                ", pageSize=" + pageSize +
                ", totalRecord=" + totalRecord +
                ", totalPage=" + totalPage +
                ", startIndex=" + startIndex +
                ", list=" + list +
                ", start=" + start +
                ", end=" + end +
                '}';
    }
}
