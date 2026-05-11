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
from collections import defaultdict
import torch
import random
import numpy as np
from heuristics.babsr import BabsrBranching
from heuristics.fsb import FsbBranching
from utils import get_reduce_op
from heuristics.utils import compute_ratio


class LookaheadBranching(BabsrBranching):

    def __init__(self, net, lookahead_discount=0.5, adaptive_rounds=5):
        super().__init__(net)
        self.lookahead_discount = lookahead_discount
        # Cache frequently accessed data
        self.split_node_names = {i: net.split_nodes[i].name for i in range(len(net.split_nodes))}
        self.name_to_idx = {name: i for i, name in self.split_node_names.items()}
        # Pre-allocate dictionary for alpha scores
        self.alpha_scores = {}
        self.alpha_score_batch = None  # For batch vectorization
        # Track branch and bound rounds for adaptive strategy
        self.adaptive_rounds = adaptive_rounds
        self.current_round = 0
        self.fsb_brancher = FsbBranching(net)

    @torch.no_grad()
    def get_branching_decisions(
            self, domains, split_depth, branching_candidates=15,
            branching_reduceop='min', use_beta=False, prioritize_alphas='none',
            **kwargs):
        
        # Increment round counter
        self.current_round += 1
        
        # After adaptive_rounds rounds, switch to FSB branching
        if self.current_round > self.adaptive_rounds:
            print(f"Switching to FSB branching after {self.adaptive_rounds} rounds.")
            return self.fsb_brancher.get_branching_decisions(
                domains, split_depth, branching_candidates,
                branching_reduceop, use_beta, prioritize_alphas,
                **kwargs
            )
            
        # Continue with full lookahead branching for early rounds
        lower_bounds, upper_bounds = domains['lower_bounds'], domains['upper_bounds']
        orig_mask, lAs, cs = domains['mask'], domains['lAs'], domains['cs']
        history = domains['history']
        alphas, betas = domains['alphas'], domains['betas']
        rhs = domains['thresholds']

        batch = len(next(iter(orig_mask.values())))
        
        # Add tensor caching - pre-compute device and constants
        device = next(iter(lower_bounds.values())).device
        inf_pos = torch.tensor(float('inf'), device=device)
        inf_neg = torch.tensor(float('-inf'), device=device)
        
        # Mask is 1 for unstable neurons. Otherwise it's 0.
        mask = orig_mask
        reduce_op = get_reduce_op(branching_reduceop)
        # In case number of unstable neurons less than topk
        topk = min(branching_candidates,
                   int(sum([item.sum() for item in mask.values()]).item()))
        number_bounds = 1 if cs is None else cs.shape[1]
        score, intercept_tb = self.babsr_score(
            lower_bounds, upper_bounds, lAs, mask, reduce_op,
            number_bounds, prioritize_alphas)

        # Following FSB's approach for tracking decisions and results
        final_decision = [[] for _ in range(batch)]
        decision_tmp = {}
        tmp_ret = {}
        score_from_layer_idx = 1 if len(score) > 1 else 0
        skip_layers = list(range(score_from_layer_idx))
        
        # Pre-compute random layer order for fallback selections
        random_layer_order = np.random.permutation(len(self.net.split_nodes))

        # real batch = batch * 2, since we have two kinds of scores - do this once
        rhs_extended = torch.cat([rhs, rhs])
        if cs is not None:
            cs_extended = torch.cat([cs, cs])
        else:
            cs_extended = None

        # For alphas, prepare them with views when possible:
        sps = defaultdict(dict)
        for k, vv in alphas.items():
            sps[k] = {}
            for kk, v in vv.items():
                # Use torch.cat only for the dimension that needs duplication
                sps[k][kk] = torch.cat([v, v], dim=2)

        # Handle betas similarly
        if use_beta:
            bs = [torch.cat([i, i]) for i in betas]
            history_extended = history + history
        else:
            history_extended = history
            
        set_alpha = True  # We only set the alpha once.
        
        # Process each layer for branching candidates
        for i in range(score_from_layer_idx, len(score)):
            # Skip layers with no useful scores
            if ((score[i].max(1).values <= 1e-4).all() and
                    (intercept_tb[i].min(1).values >= -1e-4).all()):
                print(f'{i}th layer has no valid scores')
                skip_layers.append(i)
                continue
            
            layer_name = self.net.split_nodes[i].name
            if layer_name not in mask or mask[layer_name].size(0) == 0:
                continue
                
            # Calculate actual topk for this layer based on number of unstable neurons
            layer_topk = min(topk, int(mask[layer_name].sum().item()))
            if layer_topk == 0:
                continue

            # Apply mask before topk - for alpha scores (max selection)
            layer_mask = mask[layer_name].bool()
            score_idx = self.masked_topk_max(score[i], layer_mask, topk)

            # Apply mask before topk - for intercept scores (min selection)
            itb_idx = self.masked_topk_min(intercept_tb[i], layer_mask, topk)
            
            score_idx_indices = score_idx.indices.cpu()
            itb_idx_indices = itb_idx.indices.cpu()
            
            # Initialize result tensor to store lookahead results
            k_ret = torch.empty(size=(topk, batch * 2), device=score[i].device)
            k_decision = []
            
            # Pre-allocate reusable tensors for branch processing
            lookahead_bonuses = torch.zeros(batch, device=score[i].device)
            alpha_score_tensor = torch.zeros(batch, device=score[i].device)
            beta_score_tensor = torch.zeros(batch, device=score[i].device)
            balanced_scores = torch.zeros(batch, device=score[i].device)
            
            for k in range(topk):
                # Check if k is within bounds for the current layer's candidates
                if k >= score_idx_indices.shape[1] or k >= itb_idx_indices.shape[1]:
                    # Skip this k if out of bounds - layer has fewer candidates than topk
                    continue
                
                # Get alpha candidates (max score)
                decision_index = score_idx_indices[:, k]
                # Add decisions with layer's idx
                decision_max_ = [[i, j.item()] for j in decision_index]
                
                # Get beta candidates (min intercept)
                decision_index = itb_idx_indices[:, k]
                decision_min_ = [[i, j.item()] for j in decision_index]
                
                # Combine alpha and beta decisions (like FSB does)
                k_decision.append(decision_max_ + decision_min_)
                
                # Create domain data for bounds computation - don't duplicate
                args_update_bounds = {
                    'lower_bounds': lower_bounds,
                    'upper_bounds': upper_bounds, 
                    'alphas': alphas if set_alpha else {},
                    'cs': cs
                }

                # Prepare for network update - duplication needed here
                lower_bounds_full = {k: torch.cat([v, v]) for k, v in lower_bounds.items()}
                upper_bounds_full = {k: torch.cat([v, v]) for k, v in upper_bounds.items()}

                args_network_bounds = {
                    'lower_bounds': lower_bounds_full,
                    'upper_bounds': upper_bounds_full,
                    'alphas': sps if set_alpha else {},
                    'cs': cs_extended
                }

                # Set up the split
                split = {'decision': k_decision[-1]}
                self.net.build_history_and_set_bounds(args_network_bounds, split)

                # Update bounds after the split
                if use_beta:
                    args_network_bounds.update({'betas': bs, 'history': history_extended})
                    k_ret_lbs = self.net.update_bounds(
                        args_network_bounds,
                        fix_interm_bounds=True, shortcut=True, beta_bias=False)
                else:
                    k_ret_lbs = self.net.update_bounds(
                        args_network_bounds, beta=False,
                        fix_interm_bounds=True, shortcut=True, beta_bias=False)
                
                # Consider the max improvement among multi bounds in one C matrix
                if k_ret_lbs.shape[0] != rhs_extended.shape[0]:
                    # This happens when network creates duplicated bounds
                    # Make sure rhs_extended matches k_ret_lbs in dimensions
                    rhs_extended_fix = torch.cat([rhs_extended] * (k_ret_lbs.shape[0] // rhs_extended.shape[0]))
                    k_ret_lbs = (k_ret_lbs - rhs_extended_fix).max(-1).values
                else:
                    k_ret_lbs = (k_ret_lbs - rhs_extended).max(-1).values
                
                # We only set alpha once
                set_alpha = False
                
                # Build mask for invalid scores (stable neurons)
                mask_score = (score_idx.values[:, k] <= 1e-4).float()
                mask_itb = (itb_idx.values[:, k] >= -1e-4).float()
                
                # Adding lookahead analysis for each candidate
                # For each branch type (alpha and beta)
                for branch_idx in range(2):
                    # Get indices for this branch type
                    start_idx = branch_idx * batch
                    end_idx = (branch_idx + 1) * batch
                    
                    # Use get_branch_data instead of manual slicing and cloning
                    branch_lbs = self.get_branch_data(args_update_bounds['lower_bounds'], branch_idx, batch)
                    branch_ubs = self.get_branch_data(args_update_bounds['upper_bounds'], branch_idx, batch)
                    
                    # Create all masks in one vectorized operation 
                    unstable_conditions = {}
                    for key in branch_lbs:
                        unstable_conditions[key] = torch.stack([branch_lbs[key] < 0, branch_ubs[key] > 0])
                    branch_mask = {k: (unstable_conditions[k].all(dim=0)).float() for k in branch_lbs}
                    
                    # Extract branch-specific alphas more efficiently
                    branch_alphas = {}
                    for key, val in args_update_bounds.get('alphas', {}).items():
                        branch_alphas[key] = {}
                        for kk, v in val.items():
                            if v.ndim > 2:
                                # Take a view instead of cloning when possible
                                branch_alphas[key][kk] = v[:, :, start_idx:end_idx]
                            else:
                                branch_alphas[key][kk] = v  # Keep shared tensors as is
                    
                    # Only clone lAs if they need to be modified
                    branch_lAs = {key: val.clone() if isinstance(val, torch.Tensor) else val for key, val in lAs.items()}
                    
                    # Extract other branch-specific data
                    branch_rhs = rhs_extended[start_idx:end_idx]
                    branch_cs = cs_extended[start_idx:end_idx] if cs_extended is not None else None
                    
                    # Create branch domains
                    branch_domains = {
                        'lower_bounds': branch_lbs,
                        'upper_bounds': branch_ubs,
                        'mask': branch_mask,
                        'lAs': branch_lAs,
                        'alphas': branch_alphas,
                        'thresholds': branch_rhs
                    }
                    
                    if branch_cs is not None:
                        branch_domains['cs'] = branch_cs
                    else:
                        branch_domains['cs'] = None
                    
                    if use_beta:
                        # Use get_branch_data for betas too
                        branch_betas = [b[start_idx:end_idx] for b in bs]
                        branch_domains['betas'] = branch_betas
                        branch_domains['history'] = history_extended[start_idx:end_idx].copy()
                    else:
                        branch_domains['betas'] = []
                        branch_domains['history'] = []
                                            
                    # Get FSB scores for this branch (lookahead)
                    fsb_scores, _, _ = self.get_scores(
                        branch_domains,
                        split_depth=1,
                        branching_candidates=1,
                        branching_reduceop=branching_reduceop,
                        use_beta=use_beta,
                        prioritize_alphas=prioritize_alphas
                    )
                    
                    # Vectorize lookahead calculations
                    branch_k_ret_lbs = k_ret_lbs[start_idx:end_idx]
                    can_verify = branch_k_ret_lbs > 0
                    
                    # Reset reused tensor instead of creating a new one
                    lookahead_bonuses.zero_()
                    
                    # Apply verification bonus
                    lookahead_bonuses[can_verify] = 1000.0
                    
                    # If we can't verify immediately, calculate second-level scores
                    if branch_idx == 0:  # Alpha branch
                        # Reset reused tensor instead of creating a new one
                        alpha_score_tensor.zero_()
                        
                        # Collect all second-level scores in a tensor
                        for fsb_layer, layer_scores in fsb_scores.items():
                            if layer_scores.size(0) > 0 and layer_scores.size(1) >= batch:
                                # Get best score per batch element across all neurons in this layer
                                layer_best_scores = layer_scores[:, :batch].max(dim=0).values
                                # Update alpha scores with better values
                                alpha_score_tensor = torch.maximum(alpha_score_tensor, layer_best_scores)
                        
                        # Store for beta branch to use
                        self.alpha_score_batch = alpha_score_tensor
                        
                        # Only apply bonus for immediate verification
                        # (balanced bonus will be added when beta branch is processed)
                    else:  # Beta branch
                        # Reset reused tensor instead of creating a new one
                        beta_score_tensor.zero_()
                        
                        # Get beta scores tensor
                        for fsb_layer, layer_scores in fsb_scores.items():
                            if layer_scores.size(0) > 0 and layer_scores.size(1) >= batch:
                                layer_best_scores = layer_scores[:, :batch].max(dim=0).values
                                beta_score_tensor = torch.maximum(beta_score_tensor, layer_best_scores)
                        
                        # Calculate balanced score vectorized: a*b/(a+b+1)
                        alpha_score_tensor = self.alpha_score_batch  # Get from previous calculation
                        numerator = alpha_score_tensor * beta_score_tensor
                        denominator = alpha_score_tensor + beta_score_tensor + 1.0
                        
                        # Use pre-allocated tensor for balanced scores
                        balanced_scores = numerator / denominator
                        
                        # Apply discount to balanced score where not immediately verified
                        not_verified = ~can_verify
                        lookahead_bonuses[not_verified] = self.lookahead_discount * balanced_scores[not_verified]
                        
                        # Also apply the balanced bonus to the alpha branch
                        k_ret[k, :batch] += self.lookahead_discount * balanced_scores
                    
                    # Apply the bonuses to the appropriate branch
                    if branch_idx == 0:  # Alpha branch
                        k_ret[k, :batch] += lookahead_bonuses
                    else:  # Beta branch
                        k_ret[k, batch:] += lookahead_bonuses
                
                # Make invalid lower bounds worse than normal lower bounds
                # Only consider the best lower bound across two splits
                # Apply the base FSB scoring after lookahead adjustments
                k_ret[k] = reduce_op((
                    k_ret_lbs.view(-1) - torch.cat(
                        [mask_score, mask_itb]).repeat(2) * 999999
                    ).reshape(2, -1),
                    dim=0).values
            
            # Update split_depth in case we have fewer candidates than requested
            split_depth = min(split_depth, k_ret.shape[0])
            
            # Select top candidates from this layer
            i_idx = k_ret.topk(split_depth, dim=0)
            tmp_ret[i] = i_idx.values  # [split_depth, batch*2]
            tmp_indice = i_idx.indices
            
            # Extract decisions corresponding to top scores with index safety check
            decision_list = []
            for ii in range(split_depth * (batch * 2)):
                row_idx = tmp_indice[ii // (2 * batch)][ii % (2 * batch)].item()
                # Check if the index is valid before accessing k_decision
                if row_idx < len(k_decision):
                    decision_list.append(k_decision[row_idx][ii % (2 * batch)])
                else:
                    # Skip this iteration if index is out of range
                    if len(k_decision) > 0:
                        # Use the first valid decision as fallback
                        decision_list.append(k_decision[0][ii % (2 * batch)])
                    else:
                        # Very rare case - create a dummy decision
                        decision_list.append([i, 0])
            
            decision_tmp[i] = decision_list

        # Process results across all layers (like FSB)
        if len(tmp_ret):
            stacked_layers = torch.stack([i for i in tmp_ret.values()])  # [layer, split_depth, batch*2]
            max_ret = torch.topk(stacked_layers.view(-1, batch * 2), split_depth,
                                 dim=0)  # compare across layers [split_depth, batch*2]
            
            # Get values and decision layers
            rets, decision_layers = max_ret.values.view(-1).cpu().numpy(), max_ret.indices.view(
                -1).cpu().numpy()  # first batch: score; second batch: intercept_tb.
            decision_layers = decision_layers // split_depth

            # Create a mapping between adjusted and unadjusted layer indices
            layer_mapping = {}
            current_adjusted = 0
            
            # Build a mapping of valid layers (not in skip_layers)
            valid_layers = [i for i in range(len(self.net.split_nodes)) if i not in skip_layers]
            
            # Create mapping from decision layer index to actual layer index
            for orig_idx in valid_layers:
                layer_mapping[current_adjusted] = orig_idx
                current_adjusted += 1
                
            # Convert decision_layers to actual layer indices using our mapping
            adjusted_decision_layers = []
            for dl in decision_layers:
                if dl in layer_mapping:
                    adjusted_decision_layers.append(layer_mapping[dl])
                else:
                    # If we don't have a mapping, take the first valid layer as fallback
                    adjusted_decision_layers.append(valid_layers[0] if valid_layers else 0)
            
            # Replace decision_layers with properly adjusted indices
            decision_layers = np.array(adjusted_decision_layers)

            # Create final decisions for each batch
            for l in range(split_depth):
                for b in range(batch):
                    # Get decisions from both alpha and beta approaches - with safety checks
                    decision_layer_1_idx = decision_layers[2 * l * batch + b].item()
                    decision_layer_2_idx = decision_layers[2 * l * batch + b + batch].item()
                    
                    # Check if the indexed decision layers exist in decision_tmp
                    if decision_layer_1_idx not in decision_tmp or decision_layer_2_idx not in decision_tmp:
                        # Skip this decision and try fallback approach
                        mask_item = {k: m[b] for k, m in mask.items()}
                        len_final_decision = len(final_decision[b])
                        # Try random order fallback
                        for preferred_layer in random_layer_order:
                            preferred_layer_ = self.net.split_nodes[preferred_layer].name
                            if preferred_layer_ in mask_item and mask_item[preferred_layer_].sum() > 0:
                                nonzero_indices = mask_item[preferred_layer_].nonzero(as_tuple=False)
                                if len(nonzero_indices) > 0:
                                    final_decision[b].append([preferred_layer_, nonzero_indices[0].item()])
                                    # Mark the selected neuron as stable
                                    mask[preferred_layer_][b][final_decision[b][-1][1]] = 0
                                    break
                        continue
                    
                    # Safe access to decision indices
                    if l * 2 * batch + b < len(decision_tmp[decision_layer_1_idx]) and \
                       l * 2 * batch + b + batch < len(decision_tmp[decision_layer_2_idx]):
                        decision_layer_1, decision_index_1 = decision_tmp[decision_layer_1_idx][l * 2 * batch + b]
                        decision_layer_2, decision_index_2 = decision_tmp[decision_layer_2_idx][l * 2 * batch + b + batch]
                    else:
                        # Skip this decision due to index error and try fallback approach
                        mask_item = {k: m[b] for k, m in mask.items()}
                        len_final_decision = len(final_decision[b])
                        for preferred_layer in random_layer_order:
                            preferred_layer_ = self.net.split_nodes[preferred_layer].name
                            if preferred_layer_ in mask_item and mask_item[preferred_layer_].sum() > 0:
                                nonzero_indices = mask_item[preferred_layer_].nonzero(as_tuple=False)
                                if len(nonzero_indices) > 0:
                                    final_decision[b].append([preferred_layer_, nonzero_indices[0].item()])
                                    # Mark the selected neuron as stable
                                    mask[preferred_layer_][b][final_decision[b][-1][1]] = 0
                                    break
                        continue
                    
                    # Convert numeric indices to layer names with safety checks
                    if not (0 <= decision_layer_1 < len(self.net.split_nodes)) or \
                       not (0 <= decision_layer_2 < len(self.net.split_nodes)):
                        # Skip this decision due to invalid layer indices
                        continue
                       
                    decision_layer_1_name = self.net.split_nodes[decision_layer_1].name
                    decision_layer_2_name = self.net.split_nodes[decision_layer_2].name
                    len_final_decision = len(final_decision[b])
                    
                    # Check if the layer names exist in the mask
                    if decision_layer_1_name not in mask or decision_layer_2_name not in mask:
                        # Skip this decision and try fallback approach
                        mask_item = {k: m[b] for k, m in mask.items()}
                        for preferred_layer in random_layer_order:
                            preferred_layer_ = self.net.split_nodes[preferred_layer].name
                            if preferred_layer_ in mask_item and mask_item[preferred_layer_].sum() > 0:
                                nonzero_indices = mask_item[preferred_layer_].nonzero(as_tuple=False)
                                if len(nonzero_indices) > 0:
                                    final_decision[b].append([preferred_layer_, nonzero_indices[0].item()])
                                    # Mark the selected neuron as stable
                                    mask[preferred_layer_][b][final_decision[b][-1][1]] = 0
                                    break
                        continue
                    
                    # Check if the indices are valid for the mask dimensions
                    if decision_index_1 >= mask[decision_layer_1_name][b].shape[0] or \
                       decision_index_2 >= mask[decision_layer_2_name][b].shape[0]:
                        # Skip this decision due to invalid neuron indices
                        continue
                    
                    # Vectorize valid decision checking
                    score_valid = max([s[b].max() for s in score]) > 1e-4
                    intercept_valid = min([s[b].min() for s in intercept_tb]) < -1e-4
                    ret_valid = max(rets[2 * l * batch + b], rets[2 * l * batch + b + batch]) > -10000
                    mask_valid = (mask[decision_layer_1_name][b][decision_index_1] != 0 or 
                                 mask[decision_layer_2_name][b][decision_index_2] != 0)
                    
                    # Check if either decision is valid
                    if score_valid and intercept_valid and ret_valid and mask_valid:
                        # Choose the better decision between alpha and beta methods
                        if (rets[2 * l * batch + b] > rets[2 * l * batch + b + batch]
                                and mask[decision_layer_1_name][b][decision_index_1] != 0):  # score > intercept_tb
                            final_decision[b].append([decision_layer_1_name, decision_index_1])
                        elif mask[decision_layer_2_name][b][decision_index_2] != 0:
                            final_decision[b].append([decision_layer_2_name, decision_index_2])
                        else:
                            # Fallback to random selection if both are invalid
                            mask_item = {k: m[b] for k, m in mask.items()}
                            # Use pre-computed random order
                            for preferred_layer in random_layer_order:
                                preferred_layer_ = self.net.split_nodes[preferred_layer].name
                                if preferred_layer_ in mask_item and mask_item[preferred_layer_].sum() > 0:
                                    nonzero_indices = mask_item[preferred_layer_].nonzero(as_tuple=False)
                                    if len(nonzero_indices) > 0:
                                        final_decision[b].append([preferred_layer_, nonzero_indices[0].item()])
                                        break
                    else:
                        # Using a random choice if no valid decision
                        mask_item = {k: m[b] for k, m in mask.items()}
                        # Use pre-computed random order
                        for preferred_layer in random_layer_order:
                            preferred_layer_ = self.net.split_nodes[preferred_layer].name
                            if preferred_layer_ in mask_item and mask_item[preferred_layer_].sum() > 0:
                                nonzero_indices = mask_item[preferred_layer_].nonzero(as_tuple=False)
                                if len(nonzero_indices) > 0:
                                    final_decision[b].append([preferred_layer_, nonzero_indices[0].item()])
                                    break
                    
                    # Mark the selected neuron as stable to avoid selecting it again
                    if len(final_decision[b]) > len_final_decision:
                        final_decision_ = final_decision[b][-1][0]  # This is already the layer name string
                        decision_index_ = final_decision[b][-1][1]
                        # Check if the key exists in the mask before trying to mark it as stable
                        if final_decision_ in mask:
                            # Check if the index is valid for this layer's mask
                            if decision_index_ < mask[final_decision_][b].shape[0]:
                                mask[final_decision_][b][decision_index_] = 0
                            else:
                                print(f"Warning: Index {decision_index_} out of bounds for layer {final_decision_} mask with shape {mask[final_decision_][b].shape}")
                        else:
                            print(f"Warning: Layer {final_decision_} not found in mask dictionary")
        else:
            # All layers are split or have no improvement - use fallback approach
            for l in range(split_depth):
                for b in range(batch):
                    mask_item = {k: m[b] for k, m in mask.items()}
                    len_final_decision = len(final_decision[b])
                    # Try from last layer to first (deeper layers preferred)
                    for preferred_layer in range(len(self.net.split_nodes)-1, -1, -1):
                        preferred_layer_ = self.net.split_nodes[preferred_layer].name
                        if preferred_layer_ in mask_item and mask_item[preferred_layer_].sum() > 0:
                            nonzero_indices = mask_item[preferred_layer_].nonzero(as_tuple=False)
                            if len(nonzero_indices) > 0:
                                final_decision[b].append([preferred_layer_, nonzero_indices[0].item()])
                                final_decision_ = preferred_layer_  # Already the layer name
                                # Check if the key exists in the mask before trying to mark it as stable
                                if final_decision_ in mask:
                                    mask[final_decision_][b][final_decision[b][-1][1]] = 0
                                else:
                                    print(f"Warning: Layer {final_decision_} not found in mask dictionary")
                                break

        # Get the minimum number of decisions across all batch elements
        split_depth = min([len(d) for d in final_decision])
        
        # Reformat final_decision to match expected output format
        final_decision = [[batch[i] for batch in final_decision] for i in
                          range(split_depth)]  # change the order to split_depth * batch
        final_decision = sum(final_decision, [])

        # Convert layer names back to numeric indices for compatibility with printing functions
        # The utils.py printing code expects numeric indices, not layer names
        for i in range(len(final_decision)):
            if isinstance(final_decision[i][0], str):
                # If first element is a string (layer name), convert it to layer index
                layer_name = final_decision[i][0]
                # Find the index of the layer with this name
                for j, node in enumerate(self.net.split_nodes):
                    if node.name == layer_name:
                        final_decision[i][0] = j
                        break

        return final_decision, None, split_depth  # None for points

    @torch.no_grad()
    def get_scores(
            self, domains, split_depth, branching_candidates=1,
            branching_reduceop='min', use_beta=False, prioritize_alphas='none',
            **kwargs):
        
        lower_bounds, upper_bounds = domains['lower_bounds'], domains['upper_bounds']
        orig_mask, lAs, cs = domains['mask'], domains['lAs'], domains['cs']
        history = domains['history'] if 'history' in domains else []
        alphas = domains['alphas'] if 'alphas' in domains else {}
        betas = domains['betas'] if 'betas' in domains else []
        rhs = domains['thresholds']
        
        batch = len(next(iter(orig_mask.values())))
        # Mask is 1 for unstable neurons. Otherwise it's 0.
        mask = orig_mask
        reduce_op = get_reduce_op(branching_reduceop)
        
        # In case number of unstable neurons less than topk
        total_unstable = sum([item.sum() for item in mask.values()]).item()
        if total_unstable == 0:
            # No unstable neurons to score
            return {}, [], []
            
        topk = min(branching_candidates, int(total_unstable))
        number_bounds = 1 if cs is None else cs.shape[1]
        
        # Approximate BaBSR scoring without needing updated lAs
        layer_scores = {}
        score_from_layer_idx = 1 if len(self.net.split_nodes) > 1 else 0
        
        # Pre-compute layer depth factors for all layers
        layer_depth_factors = torch.tensor([
            (i - score_from_layer_idx + 1) / (len(self.net.split_nodes) - score_from_layer_idx)
            for i in range(score_from_layer_idx, len(self.net.split_nodes))
        ], device=rhs.device)
        
        # Pre-calculate bound metrics for all layers
        bound_widths = {}
        activation_closeness = {}
        zero_positions = {}
        
        for input_name in lower_bounds:
            if input_name in mask:
                l_bound = lower_bounds[input_name]
                u_bound = upper_bounds[input_name]
                
                # Calculate metrics in one pass
                bound_widths[input_name] = u_bound - l_bound
                min_abs = torch.min(u_bound.abs(), l_bound.abs())
                activation_closeness[input_name] = torch.clamp(
                    1.0 - min_abs / (bound_widths[input_name] + 1e-8), 0.0, 1.0)
                zero_positions[input_name] = torch.clamp(
                    -l_bound / (bound_widths[input_name] + 1e-8), 0.0, 1.0)
        
        # Track approximated alpha and intercept scores
        alpha_scores = []
        intercept_scores = []
        
        # Create approximated scores for each layer
        for i in range(score_from_layer_idx, len(self.net.split_nodes)):
            layer = self.net.split_nodes[i]
            key = layer.name
            input_name = layer.inputs[0].name if hasattr(layer, 'inputs') and len(layer.inputs) > 0 else None
            
            # Skip if no unstable neurons
            if input_name is None or input_name not in mask:
                alpha_scores.append(torch.zeros((batch, 1), device=rhs.device))
                intercept_scores.append(torch.zeros((batch, 1), device=rhs.device))
                continue
                
            # Get bounds for this layer
            l_bound = lower_bounds[input_name]
            u_bound = upper_bounds[input_name]
            
            # The mask for unstable neurons
            this_mask = mask[input_name]
            
            # Skip if no unstable neurons
            if this_mask.sum() == 0:
                alpha_scores.append(torch.zeros((batch, this_mask.shape[1]), device=rhs.device))
                intercept_scores.append(torch.zeros((batch, this_mask.shape[1]), device=rhs.device))
                continue
            
            # Calculate ratio terms similar to BaBSR
            ratio_0, ratio_1 = compute_ratio(l_bound, u_bound)
            
            # Use pre-computed metrics
            bound_width = bound_widths[input_name]
            activation_close = activation_closeness[input_name]
            zero_position = zero_positions[input_name]
            
            # Create an approximation of gradients vectorized
            layer_depth_factor = layer_depth_factors[i - score_from_layer_idx]
            
            # Scale the gradient by a measure of instability
            instability_factor = 4.0 * zero_position * (1.0 - zero_position)  # Peaks at 0.5
            approx_gradient = bound_width * activation_close * layer_depth_factor * instability_factor
            
            # Create approximate BaBSR score components
            intercept_approx = -approx_gradient * ratio_1 * this_mask
            
            # Approximate the bias terms
            bias_candidate_1 = approx_gradient * (ratio_0 - 1)
            bias_candidate_2 = approx_gradient * ratio_0
            
            # Combine using the reduce operator (typically max for BaBSR)
            bias_candidate = reduce_op(bias_candidate_1, bias_candidate_2)
            alpha_approx = (bias_candidate.abs() + intercept_approx.abs()) * this_mask
            
            # Store the approximated scores
            alpha_scores.append(alpha_approx)
            intercept_scores.append(intercept_approx)
            
            # For each layer, get topk neurons based on alpha score
            if alpha_approx.sum() > 0:
                k_ret = torch.zeros(size=(topk, batch * 2), device=alpha_approx.device)
                
                # Process alpha scores
                alpha_values, alpha_indices = torch.topk(alpha_approx, 
                                                     min(topk, int(this_mask.sum())), 
                                                     dim=1)
                
                # Process intercept scores (prefer more negative values)
                intercept_values, intercept_indices = torch.topk(
                    -intercept_approx,  # Negate to use topk for minimization 
                    min(topk, int(this_mask.sum())), 
                    dim=1)
                intercept_values = -intercept_values  # Restore original sign
                
                # Fill scores for both branches for each topk candidate in batches
                max_k = min(topk, alpha_values.shape[1])
                
                # Fill alpha scores (first half of batch)
                for k in range(max_k):
                    k_ret[k, :batch] = alpha_values[:, k]
                
                # Fill intercept scores (second half of batch)
                for k in range(min(topk, intercept_values.shape[1])):
                    k_ret[k, batch:] = intercept_values[:, k].abs()  # Make comparable with alpha scores
                
                layer_scores[i] = k_ret
        
        return layer_scores, alpha_scores, intercept_scores

    def get_branch_data(self, tensor_dict, branch_idx, batch_size):
        """Access original tensors with indexing rather than creating duplicates."""
        start_idx = branch_idx * batch_size
        end_idx = (branch_idx + 1) * batch_size
        return {k: v[start_idx:end_idx] for k, v in tensor_dict.items()}

    # For max selection (larger values preferred)
    def masked_topk_max(self, scores, mask, k, dim=1):
        # Avoid cloning by using torch.where
        masked_scores = torch.where(
            mask.bool(), 
            scores, 
            torch.tensor(float('-inf'), device=scores.device, dtype=scores.dtype)
        )
        return torch.topk(masked_scores, min(k, int(mask.sum().item())), dim=dim)

    # For min selection (smaller values preferred)
    def masked_topk_min(self, scores, mask, k, dim=1):
        # Avoid cloning by using torch.where
        masked_scores = torch.where(
            mask.bool(), 
            scores, 
            torch.tensor(float('inf'), device=scores.device, dtype=scores.dtype)
        )
        return torch.topk(masked_scores, min(k, int(mask.sum().item())), dim=dim, largest=False)
