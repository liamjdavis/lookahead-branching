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


def set_alpha_input_split(self, alpha, set_all=False):
    if len(alpha) == 0:
        return
    for m in self.net.perturbed_optimizable_activations:
        for spec_name in list(m.alpha.keys()):
            if spec_name in alpha[m.name]:
                # Only setup the last layer alphas if no refinement is done.
                if spec_name in self.alpha_start_nodes or set_all:
                    # Copy alpha. Size is (2, spec, batch, *shape).
                    m.alpha[spec_name] = alpha[m.name][spec_name].detach().requires_grad_()
            else:
                # This layer's alpha is not used. For example, we can drop all intermediate layer alphas.
                del m.alpha[spec_name]
