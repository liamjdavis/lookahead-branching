#########################################################################
##   This file is part of the α,β-CROWN (alpha-beta-CROWN) verifier    ##
##                                                                     ##
##   Copyright (C) 2021-2025 The α,β-CROWN Team                        ##
##   Primary contacts: Huan Zhang <huan@huan-zhang.com> (UIUC)         ##
##                     Zhouxing Shi <zshi@cs.ucla.edu> (UCLA)          ##
##                     Xiangru Zhong <xiangru4@illinois.edu> (UIUC)    ##
##                                                                     ##
##    See CONTRIBUTORS for all author contacts and affiliations.       ##
##                                                                     ##
##     This program is licensed under the BSD 3-Clause License,        ##
##        contained in the LICENCE file in this directory.             ##
##                                                                     ##
#########################################################################
import math
import torch
from torch import Tensor
from typing import Union, Tuple, Optional

import arguments


@torch.no_grad()
def input_split_parallel(
        x_L: Tensor,
        x_U: Tensor,
        shape: tuple = None,
        split_depth: int = 1,
        i_idx: Optional[Tensor] = None,
        split_partitions: int = 2,
        split_hint: Optional[list[float]] = None
) -> Tuple[Tensor, Tensor, int]:
    """
    Split the x_L and x_U given split_idx and split_depth.

    :param x_L:                 The lower bound on the inputs of the subdomains
    :param x_U:                 The upper bound on the inputs of the subdomains
    :param shape:
    :param split_depth:         How many splits we wish to consider for all subdomains where split_depth <= input_dim
    :param i_idx:               Input indices to split on for each batch
    :param split_partitions:    Partitions per node. split_partition=2 simply creates a binary tree.
    :param split_hint:          Only valid when split_partitions=2, the domains get split at split_hint rather than the
                                midpoint
                                (lb + ub)/2. This is beneficial when clipping domains.

    :return new_x_L:            The lower bound on the inputs of the new subdomains
    :return new_x_U:            The upper bound on the inputs of the new subdomains
    :return split_depth:        The depth of the split    
    """
    # FIXME: this function should not be in this file.
    x_L = x_L.flatten(1)
    x_U = x_U.flatten(1)

    x_L_cp = x_L.clone()
    x_U_cp = x_U.clone()

    split_depth = min(split_depth, i_idx.size(1))
    remaining_depth = split_depth
    input_dim = x_L.shape[1]

    if split_hint is not None:
        assert split_partitions == 2, "Can only handle split_hint with split_partitions==2"
        # convert split_hint to a tensor with length input_dim
        if len(split_hint) == 1:
            split_hint = torch.tensor(split_hint, device=x_L_cp.device, dtype=x_L_cp.dtype).expand(input_dim)
        else:
            assert len(split_hint) == input_dim, f"split_dim has dimension {len(split_hint)} when input_dim is {input_dim}"
            split_hint = torch.tensor(split_hint, device=x_L_cp.device, dtype=x_L_cp.dtype)

    while remaining_depth > 0:
        for i in range(min(input_dim, remaining_depth)):
            indices = torch.arange(x_L_cp.shape[0])
            copy_num = x_L_cp.shape[0]//x_L.shape[0]
            idx = i_idx[:,i].repeat(copy_num).long()

            has_crossing = None
            if split_hint is not None:
                # find dimensions that have a crossing at split_hint
                has_crossing = torch.logical_and(x_L_cp[indices, idx] < split_hint[idx], x_U_cp[indices, idx] > split_hint[idx])

            x_L_cp_list, x_U_cp_list = [], []
            for partition in range(split_partitions):
                x_L_cp_tmp = x_L_cp.clone()
                x_U_cp_tmp = x_U_cp.clone()

                lrange = ((partition + 1) * x_L_cp[indices, idx] +
                          (split_partitions - partition - 1) * x_U_cp[indices, idx]) / split_partitions
                urange = (partition * x_L_cp[indices, idx] +
                          (split_partitions - partition) * x_U_cp[indices, idx]) / split_partitions

                if split_hint is not None:
                    if partition == 0:
                        # creating upper subdomains with split_hint
                        x_L_cp_tmp[indices, idx] = torch.where(has_crossing, split_hint[idx], lrange)
                        x_U_cp_tmp[indices, idx] = urange
                    else:
                        # creating lower subdomains with split_hint
                        x_L_cp_tmp[indices, idx] = lrange
                        x_U_cp_tmp[indices, idx] = torch.where(has_crossing, split_hint[idx], urange)
                else:
                    x_L_cp_tmp[indices, idx] = lrange
                    x_U_cp_tmp[indices, idx] = urange

                x_L_cp_list.append(x_L_cp_tmp)
                x_U_cp_list.append(x_U_cp_tmp)

            x_L_cp = torch.cat(x_L_cp_list)
            x_U_cp = torch.cat(x_U_cp_list)

        remaining_depth -= min(input_dim, remaining_depth)

    split_depth = split_depth - remaining_depth

    new_x_L = x_L_cp.reshape(-1, *shape[1:])
    new_x_U = x_U_cp.reshape(-1, *shape[1:])

    return new_x_L, new_x_U, split_depth


def get_split_depth(x_L, split_partitions=2):
    split_depth = 1
    min_batch_size_ratio = arguments.Config["solver"]["min_batch_size_ratio"]
    batch_size = arguments.Config["solver"]["batch_size"]
    if len(x_L) < min_batch_size_ratio * batch_size:
        min_batch_size = min_batch_size_ratio * batch_size
        split_depth = int(math.log(min_batch_size//len(x_L))//math.log(split_partitions))
        split_depth = max(split_depth, 1)
    return split_depth


def repeat_data_after_split(split_depth: int,
                            split_partitions: int,      
                            cs: Optional[Tensor] = None,
                            thresholds: Optional[Tensor] = None,
                            dm_lb: Optional[Tensor] = None,
                            alphas: Optional[dict] = None,
                            lA: Optional[Tensor] = None,
                            lbias: Optional[Tensor] = None):
    """
    Repeat the data after splitting the input domain.
    
    :param split_depth:         The depth of the split
    :param split_partitions:    Partitions per node. split_partition=2 simply creates a binary tree.
    :param cs:                  The C matrices of input domains
    :param thresholds:          The thresholds of input domains
    :param dm_lb:               The lower bounds of input domains
    :param alphas:              The alpha dict of the input domains
    :param lA:                  The lA matrices of the input domains
    :param lbias:               The lbias values of the input domains

    :return repeated_cs:        The repeated cs
    :return repeated_thresholds: The repeated thresholds
    :return repeated_dm_lb:     The repeated dm_lb
    :return repeated_alphas:    The repeated alphas
    :return repeated_lA:        The repeated lA
    :return repeated_lbias:     The repeated lbias

    """
    new_batch_size = split_partitions ** split_depth
    def repeat_one_dim(data: Optional[Tensor], target_dim, repeat_count):
        if data is not None:
            return data.repeat(*(1 if i != target_dim else repeat_count for i in range(data.dim())))
        return None

    repeated_cs = repeat_one_dim(cs, 0, new_batch_size)
    repeated_thresholds = repeat_one_dim(thresholds, 0, new_batch_size)
    repeated_dm_lb = repeat_one_dim(dm_lb, 0, new_batch_size)
    repeated_alphas = None
    if alphas is not None:
        repeated_alphas = {}
        for m in alphas:
            repeated_alphas[m] = {}
            for spec_name in alphas[m]:
                repeated_alphas[m][spec_name] = repeat_one_dim(alphas[m][spec_name], 2, new_batch_size)
    repeated_lA = repeat_one_dim(lA, 0, new_batch_size)
    repeated_lbias = repeat_one_dim(lbias, 0, new_batch_size)
    return repeated_cs, repeated_thresholds, repeated_dm_lb, repeated_alphas, repeated_lA, repeated_lbias

def input_split_and_repeat(
        x_L: Tensor,
        x_U: Tensor,
        shape: tuple = None,
        split_depth: int = 1,
        i_idx: Optional[Tensor] = None,
        split_partitions: int = 2,
        split_hint: Optional[list[float]] = None,
        cs: Optional[Tensor] = None,
        thresholds: Optional[Tensor] = None,
        dm_lb: Optional[Tensor] = None,
        alphas: Optional[dict] = None,
        lA: Optional[Tensor] = None,
        lbias: Optional[Tensor] = None
):
    """
    Split the input domain and repeat the data.
    This function is the wrapped version of input_split_parallel and repeat_data_after_split.

    :param x_L:                 The lower bound on the inputs of the subdomains
    :param x_U:                 The upper bound on the inputs of the subdomains
    :param shape:
    :param split_depth:         How many splits we wish to consider for all subdomains where split_depth <= input_dim
    :param i_idx:               Input indices to split on for each batch
    :param split_partitions:    Partitions per node. split_partition=2 simply creates a binary tree.
    :param split_hint:          Only valid when split_partitions=2, the domains get split at split_hint rather than the
                                midpoint
                                (lb + ub)/2. This is beneficial when clipping domains.
    :param cs:                  The C matrices of input domains
    :param thresholds:          The thresholds of input domains
    :param dm_lb:               The lower bounds of input domains
    :param alphas:              The alpha dict of the input domains
    :param lA:                  The lA matrices of the input domains
    :param lbias:               The lbias values of the input domains
                                
    :return new_x_L:            The lower bound on the inputs of the new subdomains
    :return new_x_U:            The upper bound on the inputs of the new subdomains
    :return split_depth:        The depth of the split
    :return repeated_cs:        The repeated cs
    :return repeated_thresholds: The repeated thresholds
    :return repeated_dm_lb:     The repeated dm_lb
    :return repeated_alphas:    The repeated alphas
    :return repeated_lA:        The repeated lA
    :return repeated_lbias:     The repeated lbias
    """
    new_x_L, new_x_U, split_depth = input_split_parallel(x_L, x_U, shape, split_depth, i_idx,
                                                          split_partitions, split_hint)
    repeated_cs, repeated_thresholds, repeated_dm_lb, repeated_alphas, repeated_lA, repeated_lbias = \
        repeat_data_after_split(split_depth, split_partitions, cs, thresholds, dm_lb, alphas, lA, lbias)
    return new_x_L, new_x_U, split_depth, repeated_cs, repeated_thresholds, repeated_dm_lb, repeated_alphas, repeated_lA, repeated_lbias
