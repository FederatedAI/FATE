#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys
import timeit


class Communicator:
    """
    Abstract class defining the functions that a Communicator should implement.
    """

    @classmethod
    def is_initialized(cls):
        """Returns whether the communicator has been initialized"""
        raise NotImplementedError("is_initialized is not implemented")

    @classmethod
    def get(cls):
        """Returns an instance of the communicator"""
        raise NotImplementedError("get is not implemented")

    @classmethod
    def initialize(cls, **kwargs):
        """Initializes the communicator. Call this function before using it."""
        raise NotImplementedError("initialize is not implemented")

    @classmethod
    def shutdown(cls):
        raise NotImplementedError("shutdown is not implemented")

    def send(self, tensor, dst):
        """Sends the specified tensor to the destination dst."""
        raise NotImplementedError("send is not implemented")

    def recv(self, tensor, src=None):
        """Receives a tensor from an (optional) source src."""
        raise NotImplementedError("recv is not implemented")

    def scatter(self, scatter_list, src, size=None, async_op=False):
        """Scatters a list of tensors to all parties."""
        raise NotImplementedError("scatter is not implemented")

    def reduce(self, tensor, op=None, async_op=False):
        """Reduces the tensor data across all parties."""
        raise NotImplementedError("tensor is not implemented")

    def all_reduce(self, tensor, op=None, async_op=False):
        """Reduces the tensor data across all parties; all get the final result."""
        raise NotImplementedError("tensor is not implemented")

    def gather(self, tensor, dst, async_op=False):
        """Gathers a list of tensors in a single party."""
        raise NotImplementedError("gather is not implemented")

    def all_gather(self, tensor, async_op=False):
        """Gathers tensors from all parties in a list."""
        raise NotImplementedError("all_gather is not implemented")

    def broadcast(self, tensor, src, async_op=False):
        """Broadcasts the tensor to all parties."""
        raise NotImplementedError("broadcast is not implemented")

    def barrier(self):
        """Synchronizes all processes.

        This collective blocks processes until the whole group enters this
        function.
        """
        raise NotImplementedError("barrier is not implemented")

    def send_obj(self, obj, dst):
        """Sends the specified object to the destination `dst`."""
        raise NotImplementedError("send_obj is not implemented")

    def recv_obj(self, src):
        """Receives a tensor from a source src."""
        raise NotImplementedError("recv_obj is not implemented")

    def broadcast_obj(self, obj, src):
        """Broadcasts a given object to all parties."""
        raise NotImplementedError("broadcast_obj is not implemented")

    def get_world_size(self):
        """Returns the size of the world."""
        raise NotImplementedError("get_world_size is not implemented")

    def get_rank(self):
        """Returns the rank of the current process."""
        raise NotImplementedError("get_rank is not implemented")

    def set_name(self):
        """Sets the party name of the current process."""
        raise NotImplementedError("set_name is not implemented")

    def get_name(self):
        """Returns the party name of the current process."""
        raise NotImplementedError("get_name is not implemented")

    def reset_communication_stats(self):
        """Resets communication statistics."""
        self.comm_rounds = 0
        self.comm_bytes = 0
        self.comm_time = 0

    def print_communication_stats(self):
        """
        Prints communication statistics.

        NOTE: Each party performs its own logging of communication, so one needs
        to sum the number of bytes communicated over all parties and divide by
        two (to prevent double-counting) to obtain the number of bytes
        communicated in the overall system.
        """
        import crypten

        crypten.log("====Communication Stats====")
        crypten.log("Rounds: {}".format(self.comm_rounds))
        crypten.log("Bytes: {}".format(self.comm_bytes))
        crypten.log("Communication time: {}".format(self.comm_time))

    def get_communication_stats(self):
        """
        Returns communication statistics in a Python dict.

        NOTE: Each party performs its own logging of communication, so one needs
        to sum the number of bytes communicated over all parties and divide by
        two (to prevent double-counting) to obtain the number of bytes
        communicated in the overall system.
        """
        return {
            "rounds": self.comm_rounds,
            "bytes": self.comm_bytes,
            "time": self.comm_time,
        }

    def _log_communication(self, nelement):
        """Updates log of communication statistics."""
        self.comm_rounds += 1
        self.comm_bytes += nelement * self.BYTES_PER_ELEMENT

    def _log_communication_time(self, comm_time):
        self.comm_time += comm_time


def _logging(func):
    """
    Decorator that performs logging of communication statistics.

    NOTE: Each party performs its own logging of communication, so one needs to
    sum the number of bytes communicated over all parties and divide by two
    (to prevent double-counting) to obtain the number of bytes communicated in
    the overall system.
    """
    from functools import wraps

    @wraps(func)
    def logging_wrapper(self, *args, **kwargs):
        from fate.arch.tensor.mpc.config import cfg

        # TODO: Replace this
        # - hacks the inputs into some of the functions for world_size 1:
        world_size = self.get_world_size()
        if world_size < 2:
            if func.__name__ in ["gather", "all_gather"]:
                return [args[0]]
            elif len(args) > 0:
                return args[0]

        # only log communication if needed:
        if cfg.communicator.verbose:
            rank = self.get_rank()
            _log = self._log_communication

            # count number of bytes communicates for each MPI collective:
            if func.__name__ == "barrier":
                _log(0)
            elif func.__name__ in ["send", "recv", "isend", "irecv"]:
                _log(args[0].nelement())  # party sends or receives tensor
            elif func.__name__ == "scatter":
                if args[1] == rank:  # party scatters P - 1 tensors
                    nelements = sum(x.nelement() for idx, x in enumerate(args[0]) if idx != rank)
                    _log(nelements)  # NOTE: We deal with other parties later
            elif func.__name__ == "all_gather":
                _log(2 * (world_size - 1) * args[0].nelement())
                # party sends and receives P - 1 tensors
            elif func.__name__ == "send_obj":
                nbytes = sys.getsizeof(args[0])
                _log(nbytes / self.BYTES_PER_ELEMENT)  # party sends object
            elif func.__name__ == "broadcast_obj":
                nbytes = sys.getsizeof(args[0])
                _log(nbytes / self.BYTES_PER_ELEMENT * (world_size - 1))
                # party sends object to P - 1 parties
            elif func.__name__ in ["broadcast", "gather", "reduce"]:
                multiplier = world_size - 1 if args[1] == rank else 1
                # broadcast: party sends tensor to P - 1 parties, or receives 1 tensor
                # gather: party receives P - 1 tensors, or sends 1 tensor
                # reduce: party receives P - 1 tensors, or sends 1 tensor
                if "batched" in kwargs and kwargs["batched"]:
                    nelements = sum(x.nelement() for x in args[0])
                    _log(nelements * multiplier)
                else:
                    _log(args[0].nelement() * multiplier)
            elif func.__name__ == "all_reduce":
                # each party sends and receives one tensor in ring implementation
                if "batched" in kwargs and kwargs["batched"]:
                    nelements = sum(2 * x.nelement() for x in args[0])
                    _log(nelements)
                else:
                    _log(2 * args[0].nelement())

            # execute and time the MPI collective:
            tic = timeit.default_timer()
            result = func(self, *args, **kwargs)
            toc = timeit.default_timer()
            self._log_communication_time(toc - tic)

            # for some function, we only know the object size now:
            if func.__name__ == "scatter" and args[1] != rank:
                _log(result.nelement())  # party receives 1 tensor
            if func.__name__ == "recv_obj":
                _log(sys.getsizeof(result) / self.BYTES_PER_ELEMENT)
                # party receives 1 object

            return result

        return func(self, *args, **kwargs)

    return logging_wrapper
