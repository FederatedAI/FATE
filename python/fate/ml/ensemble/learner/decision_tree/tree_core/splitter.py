#
#  Copyright 2019 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import torch
import numpy as np
import logging
from fate.arch.dataframe import DataFrame
from fate.arch import Context
from fate.arch.histogram import DistributedHistogram


logger = logging.getLogger(__name__)


# tree decimal round to prevent float error
TREE_DECIMAL_ROUND = 10


class SplitInfo(object):
    def __init__(
        self,
        best_fid=None,
        best_bid=None,
        sum_grad=0,
        sum_hess=0,
        gain=None,
        missing_dir=1,
        split_id=None,
        sample_count=-1,
        sitename=None,
    ):
        self.best_fid = best_fid
        self.best_bid = best_bid
        self.sum_grad = sum_grad
        self.sum_hess = sum_hess
        self.gain = gain
        self.missing_dir = missing_dir
        self.split_id = split_id
        self.sample_count = sample_count
        self.sitename = sitename

    def __str__(self):
        return (
            "(fid {} bid {}, sum_grad {}, sum_hess {}, gain {}, missing dir {}, split_id {}, "
            "sample_count {}, sitename {})\n".format(
                self.best_fid,
                self.best_bid,
                self.sum_grad,
                self.sum_hess,
                self.gain,
                self.missing_dir,
                self.split_id,
                self.sample_count,
                self.sitename,
            )
        )

    def __repr__(self):
        return self.__str__()


class Splitter(object):
    def __init__(self) -> None:
        pass


class SBTSplitter(Splitter):
    def __init__(
        self,
        bin_train_data: DataFrame,
        bin_info: dict,
        min_impurity_split=1e-2,
        min_sample_split=2,
        min_leaf_node=1,
        min_child_weight=1,
        l1=0,
        l2=0.1,
    ) -> None:
        super().__init__()
        self.min_impurity_split = min_impurity_split
        self.min_sample_split = min_sample_split
        self.min_leaf_node = min_leaf_node
        self.min_child_weight = min_child_weight
        self.bin_info = bin_info
        self.l1, self.l2 = l1, l2
        columns = bin_train_data.schema.columns
        self.feat_bin_num = [len(bin_info[feat]) for feat in columns]

    def get_bucket(self, idx):
        feature_buckets = self.feat_bin_num
        cumulative_buckets = [0]
        for bucket in feature_buckets:
            cumulative_buckets.append(cumulative_buckets[-1] + bucket)

        for i in range(1, len(cumulative_buckets)):
            if idx < cumulative_buckets[i]:
                fid = i - 1
                bid = idx - cumulative_buckets[i - 1]
                return fid, bid

        raise ValueError("idx is out of range")

    @staticmethod
    def truncate(f, n=TREE_DECIMAL_ROUND):
        return np.floor(f * 10**n) / 10**n

    def _l1_reg(self, g):
        if self.l1 == 0:
            return g
        if isinstance(g, torch.Tensor):
            g[g < -self.l1] += self.l1
            g[g > self.l1] -= self.l1
            g[(g <= self.l1) & (g >= -self.l1)] = 0
        else:
            if g < -self.l1:
                return g + self.l1
            elif g > self.l1:
                return g - self.l1
            else:
                return 0
        return g

    def node_gain(self, g, h):
        g, h = self.truncate(g), self.truncate(h)
        g = self._l1_reg(g)
        if isinstance(h, np.ndarray):
            h[h == 0] = np.nan
        score = (g * g) / (h + self.l2)
        return score

    def node_weight(self, sum_grad, sum_hess):
        sum_grad = self._l1_reg(sum_grad)
        weight = -(sum_grad / (sum_hess + self.l2))
        return self.truncate(weight)

    def _extract_hist(self, histogram, pack_info=None):
        tensor_hist: dict = histogram.extract_data()
        g_all, h_all, cnt_all = None, None, None
        for k, v in tensor_hist.items():
            cnt = v["cnt"].reshape((1, -1))

            # if gh pack
            if "gh" in v:
                g = v["gh"][::, 0].reshape((1, -1))
                h = v["gh"][::, 1].reshape((1, -1))
                if pack_info is None:
                    raise ValueError("must provide pack info for gh packing computing")
                g = g - pack_info["g_offset"] * cnt
            else:
                g = v["g"].reshape((1, -1))
                h = v["h"].reshape((1, -1))

            if g_all is None:
                g_all = g
            else:
                g_all = torch.vstack([g_all, g])
            if h_all is None:
                h_all = h
            else:
                h_all = torch.vstack([h_all, h])
            if cnt_all is None:
                cnt_all = cnt
            else:
                cnt_all = torch.vstack([cnt_all, cnt])

        return g_all, h_all, cnt_all

    def _make_sum_tensor(self, nodes):
        g_sum, h_sum, cnt_sum = [], [], []
        for node in nodes:
            g_sum.append(node.grad)
            h_sum.append(node.hess)
            cnt_sum.append(node.sample_num)

        return (
            torch.Tensor(g_sum).reshape((len(nodes), 1)),
            torch.Tensor(h_sum).reshape((len(nodes), 1)),
            torch.Tensor(cnt_sum).reshape((len(nodes), 1)),
        )

    def _compute_min_leaf_mask(self, l_cnt, r_cnt):
        min_leaf_node_mask_l = l_cnt < self.min_leaf_node
        min_leaf_node_mask_r = r_cnt < self.min_leaf_node
        union_mask_0 = torch.logical_or(min_leaf_node_mask_l, min_leaf_node_mask_r)
        return union_mask_0

    def _compute_gains(self, g, h, cnt, g_sum, h_sum, cnt_sum, hist_mask=None):
        l_g, l_h, l_cnt = g, h, cnt
        r_g, r_h = g_sum - l_g, h_sum - l_h
        r_cnt = cnt_sum - l_cnt

        # filter split
        # leaf count
        union_mask_0 = self._compute_min_leaf_mask(l_cnt, r_cnt)
        # min child weight
        min_child_weight_mask_l = l_h < self.min_child_weight
        min_child_weight_mask_r = r_h < self.min_child_weight
        union_mask_1 = torch.logical_or(min_child_weight_mask_l, min_child_weight_mask_r)
        if hist_mask is not None:
            mask = torch.logical_or(union_mask_0, hist_mask)
        else:
            mask = union_mask_0
        mask = torch.logical_or(mask, union_mask_1)
        rs = self.node_gain(l_g, l_h) + self.node_gain(r_g, r_h) - self.node_gain(g_sum, h_sum)
        rs = self.truncate(rs)
        rs[torch.isnan(rs)] = float("-inf")
        rs[rs < self.min_impurity_split] = float("-inf")
        rs[mask] = float("-inf")
        return rs

    def _find_best_splits(
        self, node_hist, sitename, cur_layer_nodes, reverse_node_map, recover_bucket=True, pack_info=None
    ):
        """
        recover_bucket: if node_hist is guest hist, can get the fid and bid of the split info
                        but for node_hist from host sites, histograms are shuffled, so can not get the fid and bid,
                        only hosts know them.
        """
        l_g, l_h, l_cnt = self._extract_hist(node_hist, pack_info)
        g_sum, h_sum, cnt_sum = self._make_sum_tensor(cur_layer_nodes)
        rs = self._compute_gains(l_g, l_h, l_cnt, g_sum, h_sum, cnt_sum)

        # reduce
        best = rs.max(dim=-1)
        best_gain = best[0]
        best_idx = best[1]
        logger.debug("best_idx: {}".format(best_idx))
        logger.debug("best_gain: {}".format(best_gain))

        split_infos = []
        node_idx = 0
        for idx, gain in zip(best_idx, best_gain):
            idx_ = int(idx.detach().cpu().item())
            if gain == float("-inf") or cnt_sum[node_idx] < self.min_sample_split:
                split_infos.append(None)
                logger.info("Node {} can not be further split".format(reverse_node_map[node_idx]))
            else:
                split_info = SplitInfo(
                    gain=float(gain),
                    sum_grad=float(l_g[node_idx][idx_]),
                    sum_hess=float(l_h[node_idx][idx_]),
                    sample_count=int(l_cnt[node_idx][idx_]),
                    sitename=sitename,
                )
                if recover_bucket:
                    fid, bid = self.get_bucket(idx_)
                    split_info.best_fid = fid
                    split_info.best_bid = bid
                else:
                    split_info.split_id = idx_
                split_infos.append(split_info)
            node_idx += 1

        return split_infos

    def _merge_splits(self, guest_splits, host_splits_list):
        splits = []
        for node_idx in range(len(guest_splits)):
            best_gain = -np.inf
            best_splitinfo = None
            guest_splitinfo: SplitInfo = guest_splits[node_idx]
            if guest_splitinfo is not None and guest_splitinfo.gain > best_gain:
                best_gain = guest_splitinfo.gain
                best_splitinfo = guest_splitinfo

            for host_idx in range(len(host_splits_list)):
                host_splits = host_splits_list[host_idx]
                host_splitinfo: SplitInfo = host_splits[node_idx]
                if host_splitinfo is not None and host_splitinfo.gain > best_gain:
                    best_gain = host_splitinfo.gain
                    best_splitinfo = host_splitinfo
            splits.append(best_splitinfo)

        return splits

    def _recover_pack_split(self, hist: DistributedHistogram, schema, decode_schema=None):
        host_hist = hist.decrypt(schema[0], schema[1], decode_schema)
        return host_hist

    def _local_split(self, ctx: Context, stat_rs, node_map, cur_layer_node):
        histogram = stat_rs.decrypt({}, {}, None)
        sitename = ctx.local.name
        reverse_node_map = {v: k for k, v in node_map.items()}
        local_best_splits = self._find_best_splits(
            histogram, sitename, cur_layer_node, reverse_node_map, recover_bucket=True
        )

        return local_best_splits

    def _guest_split(self, ctx: Context, stat_rs, cur_layer_node, node_map, sk, coder, gh_pack, pack_info):
        if sk is None or coder is None:
            raise ValueError("sk or coder is None, not able to decode host split points")

        histogram = stat_rs.decrypt({}, {}, None)
        sitename = ctx.local.name
        reverse_node_map = {v: k for k, v in node_map.items()}

        # find local best splits
        guest_best_splits = self._find_best_splits(
            histogram, sitename, cur_layer_node, reverse_node_map, recover_bucket=True
        )
        # find best splits from host parties
        host_histograms = ctx.hosts.get("hist")

        host_splits = []
        if gh_pack:
            decrypt_schema = ({"gh": sk}, {"gh": (coder, torch.int64)})
            # (coder, pack_num, offset_bit, precision, total_num)
            if pack_info is not None:
                decode_schema = {
                    "gh": (
                        coder,
                        pack_info["pack_num"],
                        pack_info["shift_bit"],
                        pack_info["precision"],
                        pack_info["total_pack_num"],
                    )
                }
            else:
                raise ValueError("pack info is not provided")
        else:
            decrypt_schema = ({"g": sk, "h": sk}, {"g": (coder, torch.float32), "h": (coder, torch.float32)})
            decode_schema = None

        for idx, hist in enumerate(host_histograms):
            host_sitename = ctx.hosts[idx].name
            host_hist = self._recover_pack_split(hist, decrypt_schema, decode_schema)
            # logger.debug("splitting host")
            host_split = self._find_best_splits(
                host_hist, host_sitename, cur_layer_node, reverse_node_map, recover_bucket=False, pack_info=pack_info
            )
            host_splits.append(host_split)

        # logger.debug("host splits are {}".format(host_splits))
        best_splits = self._merge_splits(guest_best_splits, host_splits)
        # logger.debug("guest splits are {}".format(guest_best_splits))
        # logger.debug("best splits are {}".format(best_splits))
        return best_splits

    def _host_split(self, ctx: Context, en_histogram, cur_layer_node):
        ctx.guest.put("hist", en_histogram)

    def split(
        self,
        ctx: Context,
        histogram_statistic_result,
        cur_layer_node,
        node_map,
        local_split=False,
        sk=None,
        coder=None,
        gh_pack=None,
        pack_info=None,
    ):
        if local_split:
            # Use local features only
            best_slits = self._local_split(ctx, histogram_statistic_result, node_map, cur_layer_node)
            return best_slits
        else:
            # For hetero-secureboost
            if ctx.is_on_guest:
                if sk is None or coder is None:
                    raise ValueError("sk or coder is None, not able to decode host split points")
                assert gh_pack is not None and isinstance(
                    gh_pack, bool
                ), "gh_pack should be bool, indicating if the gh is packed"
                if not gh_pack:
                    logger.info("not using gh pack to split")
                return self._guest_split(
                    ctx, histogram_statistic_result, cur_layer_node, node_map, sk, coder, gh_pack, pack_info
                )
            elif ctx.is_on_host:
                return self._host_split(ctx, histogram_statistic_result, cur_layer_node)
            else:
                raise ValueError("illegal role {}".format(ctx.role))
