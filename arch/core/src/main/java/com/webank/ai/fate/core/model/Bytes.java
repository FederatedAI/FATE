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

package com.webank.ai.fate.core.model;

import com.google.protobuf.ByteString;

import java.io.Serializable;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.Comparator;

public class Bytes implements Comparable<Bytes>, Serializable {

    public final static ByteArrayComparator BYTES_LEXICO_COMPARATOR = new LexicographicByteArrayComparator();
    private static final char[] HEX_CHARS_UPPER = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F'};
    private final byte[] bytes;
    private int hashCode;

    public Bytes(byte[] bytes) {
        this.bytes = bytes;

        hashCode = 0;
    }

    public static Bytes wrap(byte[] bytes) {
        if (bytes == null)
            return null;
        return new Bytes(bytes);
    }

    public static Bytes wrapUtf8String(String string) {
        if (string == null)
            return null;
        return wrap(string.getBytes(StandardCharsets.UTF_8));
    }

    public static Bytes wrap(ByteString bs) {
        if (bs == null) {
            return null;
        }
        return wrap(bs.toByteArray());
    }

    private static String toString(final byte[] b, int off, int len) {
        StringBuilder result = new StringBuilder();

        if (b == null)
            return result.toString();


        if (off >= b.length)
            return result.toString();

        if (off + len > b.length)
            len = b.length - off;

        for (int i = off; i < off + len; ++i) {
            int ch = b[i] & 0xFF;
            if (ch >= ' ' && ch <= '~' && ch != '\\') {
                result.append((char) ch);
            } else {
                result.append("\\x");
                result.append(HEX_CHARS_UPPER[ch / 0x10]);
                result.append(HEX_CHARS_UPPER[ch % 0x10]);
            }
        }
        return result.toString();
    }

    public byte[] get() {
        return this.bytes;
    }

    @Override
    public int hashCode() {
        if (hashCode == 0) {
            hashCode = Arrays.hashCode(bytes);
        }

        return hashCode;
    }

    @Override
    public boolean equals(Object other) {
        if (this == other)
            return true;
        if (other == null)
            return false;

        if (this.hashCode() != other.hashCode())
            return false;

        if (other instanceof Bytes)
            return Arrays.equals(this.bytes, ((Bytes) other).get());

        return false;
    }

    @Override
    public int compareTo(Bytes that) {
        return BYTES_LEXICO_COMPARATOR.compare(this.bytes, that.bytes);
    }

    @Override
    public String toString() {
        return Bytes.toString(bytes, 0, bytes.length);
    }

    public interface ByteArrayComparator extends Comparator<byte[]>, Serializable {

        int compare(final byte[] buffer1, int offset1, int length1,
                    final byte[] buffer2, int offset2, int length2);
    }

    private static class LexicographicByteArrayComparator implements ByteArrayComparator {

        @Override
        public int compare(byte[] buffer1, byte[] buffer2) {
            return compare(buffer1, 0, buffer1.length, buffer2, 0, buffer2.length);
        }

        public int compare(final byte[] buffer1, int offset1, int length1,
                           final byte[] buffer2, int offset2, int length2) {

            // short circuit equal case
            if (buffer1 == buffer2 &&
                    offset1 == offset2 &&
                    length1 == length2) {
                return 0;
            }

            int end1 = offset1 + length1;
            int end2 = offset2 + length2;
            for (int i = offset1, j = offset2; i < end1 && j < end2; i++, j++) {
                final int a = Byte.toUnsignedInt(buffer1[i]);
                final int b = Byte.toUnsignedInt(buffer2[j]);
                final int result = Integer.compareUnsigned(a, b);
                if (result != 0) {
                    return result;
                }
            }
            return length1 - length2;
        }
    }
}

