#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import crypten
import crypten.communicator as comm
import torch
import torch.nn as nn
from crypten.config import cfg
from crypten.gradients import _inverse_broadcast


# TODO: Move SkippedLoss elsewhere
class SkippedLoss(object):
    """Placeholder for output of a skipped loss function"""

    def __init__(self, msg=""):
        self.msg = msg

    def __repr__(self):
        return f"SkippedLoss({self.msg})"


def _matmul_backward(input, weight, grad_output):
    """Implements matmul backward from crypten.gradients

    This is necessary here because the forward in DPSplitModel is performed in plaintext
    and does not appear on the CrypTen autograd tape, so we do not have a saved ctx.

    Only returns gradient w.r.t. weight since that is all we need in this context.
    """
    # Cache sizes for inverse_broadcast
    weight_size = weight.t().size()

    # Deal with vectors that are represented by a
    # < 2 dimensional tensor
    if input.dim() < 2:
        input = input.unsqueeze(0)
        grad_output = grad_output.unsqueeze(0)

    if weight.dim() < 2:
        weight = weight.unsqueeze(1)
        grad_output = grad_output.unsqueeze(1)

    # Compute gradients
    weight_grad = input.transpose(-2, -1).matmul(grad_output)

    # Fix gradient sizes for vector inputs
    if len(weight_size) < 2:
        weight_grad = weight_grad.squeeze()
        if weight_grad.dim() < 1:
            weight_grad = weight_grad.unsqueeze(0)

    # Return weight grad
    weight_grad = _inverse_broadcast(weight_grad, weight_size).t()
    return weight_grad / weight_grad.size(0)


class DPSplitModel(nn.Module):
    """
    Differentially Private Split-MPC module that provides label-DP. Models will
    run in 6 steps:
        (1) Run forward pass in plaintext using PyTorch to get logits
        (2) Apply logistic function (sigmoid or softmax) to get predictions
        (2) Compute loss function in CrypTen
        (3) Compute dL/dZ (gradient w.r.t logits) in CrypTen
        (5) Compute aggregated parameter gradients with differential privacy
        (6) Decrypt noisy gradients

    Step (5) is computed using different methods depending on protocol configuration
        (See Config Options > protocol for descriptions)

    Args:
        pytorch_model (torch.nn.Module) : The input model to be trained
            using DP-Split-MPC algorithm. Remains in plaintext throughout.
        noise_magnitude (float) : The magnitude of DP noise to be applied to
            gradients prior to decryption for each batch of training.
        feature_src (int) : Source for input features to the model (also owns
            the plaintext model throughout training)
        label_src (int) : Source for training labels. Labels can either be input
            as plaintext values from the label_src party or as CrypTensors.

    Config Options:
        skip_loss_forward (bool) : Determines whether to compute the
            value of the loss during training (see crypten.nn._Loss definition
            of skip_forward). If True, this model will output zeros for the value
            of the loss function. However, correct gradients will still be computed
            when calling backward(). Default: True
        cache_pred_size (bool) :  Determines whether the size of the predictions should
            be cached. If True, DPSplitModel instances will remember the tensor and
            batch sizes input. This saves one communication round per batch, but
            the user will be responsible for using correct batch sizes to avoid
            crashing.
        protocol (string): Name of protocol to use to compute gradients:
                "full_jacobian": Computes the full jacobian to compute all parameter gradients from dL/dP.
                    This jacobian will be encrypted and gradients are computed by an encrypted matrix multiplication.
                "layer_estimation": Computes the jacobian only with respect to the last linear layer (dL/dW)
                    of the forward network. DP and aggregation are applied before decrypting dL/dW. This gradient
                    is then used to estimate dL/dZ (gradient w.r.t. logits). Backpropagation is
                    then computed normally in plaintext.

    Example:
        ```
        preds = dp_split_model(x)
        loss = dp_split_model.compute_loss(targets)
        dp_split_model.backward()
        ```
    """

    def __init__(
        self,
        pytorch_model,
        feature_src,
        label_src,
        noise_magnitude=None,
        noise_src=None,
        randomized_response_prob=None,
        rappor_prob=None,
    ):
        super().__init__()

        # TODO: Compute noise magnitude based on jacobian.
        self.noise_magnitude = noise_magnitude
        self.feature_src = feature_src
        self.label_src = label_src
        self.noise_src = noise_src

        # Model must be defined for model owning party
        if self.is_feature_src():
            assert isinstance(
                pytorch_model, torch.nn.Module
            ), "pytorch_model must be a torch Module"

        self.model = pytorch_model
        self.train()

        # Process Randomized Response parameters
        if randomized_response_prob is not None:
            assert (
                0 < randomized_response_prob < 0.5
            ), "randomized_response_prob must be in the interval [0, 0.5)"
        self.rr_prob = randomized_response_prob

        # Apply RAPPOR correction:
        if rappor_prob is not None:
            assert 0 <= rappor_prob <= 1, "rappor_prob must be in [0, 1]"

        self.alpha = rappor_prob

        # TODO: Add support for multi-class predictions
        self.multiclass = False

        # Cache for tensor sizes
        self.cache = {}

    def eval(self):
        self.train(mode=False)

    @property
    def training(self):
        if hasattr(self, "model") and self.model is not None:
            return self.model.training
        return self._training

    @training.setter
    def training(self, mode):
        self.train(mode)

    def train(self, mode=True):
        if hasattr(self, "model") and self.model is not None:
            self.model.train(mode=mode)
        else:
            self._training = mode

    def zero_grad(self):
        if self.is_feature_src():
            self.model.zero_grad()

    def forward(self, input):
        # During eval mode, just conduct forward pass.
        if not self.training:
            if self.is_feature_src():
                return self.model(input)
            # Parties without model should return None
            return None

        if self.is_feature_src():
            self.logits = self.model(input)
            self.preds = self.logits.sigmoid()

            # Extract saved input to last layer from autograd tape if we need it
            if cfg.nn.dpsmpc.protocol == "layer_estimation":
                self.last_input = self.logits.grad_fn._saved_mat1

            # Check that prediction size matches cached size
            preds_size = self.preds.size()
            if "preds_size" in self.cache:
                cache_size = self.cache["preds_size"]
                if preds_size != cache_size:
                    raise ValueError(
                        f"Logit size does not match cached size: {preds_size} vs. {cache_size}"
                    )

            # Cache predictions size - Note batch size must match here
            # TODO: Handle batch dimension here
            if self.cache_pred_size:
                preds_size = self._communicate_and_cache("preds_size", preds_size)
            else:
                preds_size = comm.get().broadcast_obj(preds_size, src=self.feature_src)
        else:
            # Cache predictions size - Note batch size must match here
            # TODO: Handle batch dimension here
            if self.cache_pred_size:
                preds_size = self._communicate_and_cache("preds_size", None)
            else:
                preds_size = comm.get().broadcast_obj(None, src=self.feature_src)
            self.logits = torch.empty(preds_size)
            self.preds = torch.empty(preds_size)

        return self.logits

    def _communicate_and_cache(self, name, value):
        """If the requested name is in the size_cache, return the cached size.

        On cache miss, the size will be communicated from feature_src party
        """
        # Cache hit
        if name in self.cache:
            return self.cache[name]

        # Cache miss
        value = comm.get().broadcast_obj(value, src=self.feature_src)
        self.cache[name] = value
        return value

    def is_feature_src(self):
        return self.rank == self.feature_src

    def is_label_src(self):
        return self.rank == self.label_src

    @property
    def skip_loss_forward(self):
        """Determines whether to skip the forward computation for the loss function (Default: True)"""
        return cfg.nn.dpsmpc.skip_loss_forward

    @property
    def rank(self):
        """Communicator rank in torch.distributed"""
        return comm.get().get_rank()

    @property
    def cache_pred_size(self):
        """Bool that determines whether to cache the prediction size"""
        return cfg.nn.dpsmpc.cache_pred_size

    def _process_targets(self, targets):
        """Encrypts targets and RR to targets if necessary"""
        if self.rr_prob is not None:
            flip_probs = torch.tensor(self.rr_prob).expand(targets.size())

        # Apply appropriate RR-protocol and encrypt targets if necessary
        if self.rr_prob is not None:
            flip_probs = torch.tensor(self.rr_prob).expand(targets.size())

        if crypten.is_encrypted_tensor(targets):
            if self.rr_prob is not None:
                flip_mask = crypten.bernoulli(flip_probs)
                targets = targets + flip_probs - 2 * flip_mask * targets
            targets_enc = targets
        else:
            # Label provider adds RR label flips if they are plaintext
            if self.rr_prob is not None and self.is_label_src():
                flip_mask = flip_probs.bernoulli()
                targets += flip_mask - 2 * targets * flip_mask

            # Encrypt targets:
            targets_enc = crypten.cryptensor(targets, src=self.label_src)

        return targets_enc

    def compute_loss(self, targets):
        # Process predictions and targets

        # Apply RAPPOR correction
        if self.alpha is not None:
            self.preds_rappor = self.alpha * self.preds
            self.preds_rappor += (1 - self.alpha) * (1 - self.preds)
            self.preds_enc = crypten.cryptensor(
                self.preds_rappor, src=self.feature_src, requires_grad=True
            )
        else:
            self.preds_enc = crypten.cryptensor(
                self.preds, src=self.feature_src, requires_grad=True
            )

        self.targets_enc = self._process_targets(targets)

        # Compute BCE loss or CrossEntropy loss depending on single or multiclass
        if self.skip_loss_forward:
            self.loss = SkippedLoss("Skipped CrossEntropy function")
        else:
            logits_enc = crypten.cryptensor(self.logits, src=self.feature_src)
            # BCEWithLogitsLoss
            if not self.multiclass:
                if self.alpha is None:
                    self.loss = logits_enc.binary_cross_entropy_with_logits(
                        self.targets_enc
                    )
                else:
                    self.loss = logits_enc.rappor_loss(self.targets_enc, self.alpha)
            # CrossEntropyLoss
            # TODO: Support Multi-class DPS-MPC
            else:
                raise NotImplementedError("Multi-class DPS-MPC is not supported")
                """
                if self.alpha is not None:
                    raise NotImplementedError("Multi-class RAPPOR Loss not supported")
                if self.is_feature_src:
                    logits_enc = crypten.cryptensor(self.logits, src=self.feature_src)
                else:
                    logits_enc = crypten.cryptensor(self.preds, src=self.features_src)
                self.loss = logits_enc.cross_entropy(self.targets_enc)
                """

        # Overwrite loss backward to call model's backward function:
        def backward_(self_, grad_output=None):
            self.backward(grad_output=grad_output)

        self.loss.backward = backward_

        return self.loss

    # TODO: Implement DP properly to make correct DP guarantees
    # TODO: Implement custom DP mechanism (split noise / magnitude)
    def _generate_noise_no_src(self, size):
        return crypten.randn(size) * self.noise_magnitude

    def _generate_noise_from_src(self, size):
        noise = torch.randn(size) * self.noise_magnitude
        noise = crypten.cryptensor(noise, src=self.noise_src)
        return noise

    def _add_dp_if_necessary(self, grad):
        if self.noise_magnitude is None or self.noise_magnitude == 0.0:
            return grad

        # Determine noise generation function
        generate_noise = (
            self._generate_noise_from_src
            if self.noise_src
            else self._generate_noise_no_src
        )
        noise = generate_noise(grad.size())
        with crypten.no_grad():
            grad += noise
        return grad

    def _get_last_linear_layer(self):
        layers = list(self.model.modules())
        for last_layer in reversed(layers):
            if isinstance(last_layer, torch.nn.Linear):
                break
        return last_layer

    def _compute_model_jacobians(self):
        """Compute Jacobians with respect to each model parameter

        If last_layer_only is True, this computes the jacobian only with respect
        to the parameters of the last layer of the model.
        """
        Z = self.logits.split(1, dim=-1)

        # Store partial Jacobian for each parameter
        jacobians = {}

        # dL/dW_i = sum_j (dL/dP_j * dP_j/dW_i)
        with crypten.no_grad():
            # TODO: Async / parallelize this
            for z in Z:
                z.backward(torch.ones(z.size()), retain_graph=True)

                params = self.model.parameters()

                for param in params:
                    grad = param.grad.flatten().unsqueeze(-1)

                    # Accumulate partial gradients: dL/dZ_j * dP_j/dW_i
                    if param in jacobians.keys():
                        jacobians[param] = torch.cat([jacobians[param], grad], dim=-1)
                    else:
                        jacobians[param] = grad
                    param.grad = None  # Reset grad for next p_j.backward()
        return jacobians

    def _compute_param_grads(self, dLdZ, jacobians):
        """Compute dLdW for all model parameters W"""

        # Populate parameter grad fields using Jacobians
        if self.is_feature_src():
            # Cache / communicate number of parameters
            params = torch.nn.utils.parameters_to_vector(self.model.parameters())
            num_params = params.numel()
            self._communicate_and_cache("num_params", num_params)

            # Process jacobian
            jacobian = torch.cat(
                [jacobians[param] for param in self.model.parameters()], dim=0
            )
        else:
            num_params = self._communicate_and_cache("num_params", None)
            jacobian_size = (num_params, dLdZ.size(-2))
            jacobian = torch.empty(jacobian_size)

        jacobian = crypten.cryptensor(jacobian, src=self.feature_src)

        # Compute gradeints wrt each param
        while jacobian.dim() < dLdZ.dim():
            jacobian = jacobian.unsqueeze(0)
        grad = jacobian.matmul(dLdZ)
        grad = grad.view(-1, num_params)
        grad = self._add_dp_if_necessary(grad)

        # Sum over batch dimension
        while grad.numel() != num_params:
            grad = grad.sum(0)

        # Decrypt dL/dZ_j * dZ_j/dW_i with Differential Privacy
        grads = grad.flatten().get_plain_text(dst=self.feature_src)
        return grads

    def _backward_full_jacobian(self, grad_output=None):
        """Computes backward for non-RR variant.

        To add DP noise at the aggregated gradient level,
        we compute the jacobians for dZ/dW in plaintext
        so we can matrix multiply by dL/dZ to compute our
        gradients without performing a full backward pass in
        crypten.
        """
        # Compute dL/dP_j
        dLdZ = self.preds_enc.sub(self.targets_enc).div(self.preds_enc.nelement())

        # Correct for RAPPOR Loss
        if self.alpha is not None:
            if self.is_feature_src:
                correction = 2 * self.alpha - 1
                correction *= self.preds * (1 - self.preds)
                correction /= self.preds_rappor * (1 - self.preds_rappor)
            else:
                correction = torch.empty(self.preds.size())

            correction_enc = crypten.cryptensor(correction, src=self.feature_src)
            dLdZ *= correction_enc

        # Turn batched vector into batched matrix for matmul
        dLdZ = dLdZ.unsqueeze(-1)

        # Compute Jacobians dP/dW wrt model weights
        jacobians = self._compute_model_jacobians() if self.is_feature_src() else None

        # Compute gradients dL/dW wrt model parameters
        grads = self._compute_param_grads(dLdZ, jacobians)

        # Populate grad fields of parameters:
        if self.is_feature_src():
            ind = 0
            for param in self.model.parameters():
                numel = param.numel()
                param.grad = grads[ind : ind + numel].view(param.size())
                ind += numel

    def _solve_dLdZ(self, dLdW):
        """Generates noisy dLdP using de-aggregation trick"""
        A = self.last_input
        B = dLdW

        # Apply pseudoinverse
        dLdZ = torch.linalg.lstsq(A.t(), B.t()).solution
        # dLdZ = B.matmul(A.pinverse()).t()
        return dLdZ

    def _compute_last_layer_grad(self, grad_output=None):
        # Compute dL/dP_j
        dLdZ_enc = self.preds_enc.sub(self.targets_enc).div(self.preds_enc.nelement())

        # Correct for RAPPOR Loss
        if self.alpha is not None:
            if self.is_feature_src:
                correction = 2 * self.alpha - 1
                correction *= self.preds * (1 - self.preds)
                correction /= self.preds_rappor * (1 - self.preds_rappor)
            else:
                correction = torch.empty(self.preds.size())

            correction_enc = crypten.cryptensor(correction, src=self.feature_src)
            dLdZ_enc *= correction_enc

        # Communicate / cache last layer input / weight sizes
        if self.is_feature_src():
            last_weight = self._get_last_linear_layer().weight
            self._communicate_and_cache("last_in_size", self.last_input.size())
            self._communicate_and_cache("last_weight_size", last_weight.size())

        else:
            last_in_size = self._communicate_and_cache("last_in_size", None)
            last_weight_size = self._communicate_and_cache("last_weight_size", None)
            self.last_input = torch.empty(last_in_size)
            last_weight = torch.empty(last_weight_size)

        # Encrypt last layer values
        # TODO: make this optional?
        last_input_enc = crypten.cryptensor(self.last_input, src=self.feature_src)

        # Compute last layer gradients (dLdW) and add DP if necessary
        dLdW_enc = dLdZ_enc.t().matmul(last_input_enc)
        dLdW_enc = self._add_dp_if_necessary(dLdW_enc)

        # return dLdW
        return dLdW_enc.get_plain_text(dst=self.feature_src)

    def _backward_layer_estimation(self, grad_output=None):
        with crypten.no_grad():
            # Find dLdW for last layer weights
            dLdW = self._compute_last_layer_grad(grad_output=grad_output)

        # Run backprop in plaintext
        if self.is_feature_src():
            dLdZ = self._solve_dLdZ(dLdW)
            self.logits.backward(dLdZ)

    def backward(self, grad_output=None):
        protocol = cfg.nn.dpsmpc.protocol
        with crypten.no_grad():
            if protocol == "full_jacobian":
                self._backward_full_jacobian(grad_output=grad_output)
                raise NotImplementedError(
                    "DPS protocol full_jacobian must be fixed before use."
                )
            elif protocol == "layer_estimation":
                with torch.no_grad():
                    self._backward_layer_estimation(grad_output=grad_output)
            else:
                raise ValueError(
                    f"Unrecognized DPSplitMPC backward protocol: {protocol}"
                )
