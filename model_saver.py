from __future__ import print_function, division
import re
import logging
import torch
import os

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

class ModelSaver:

    def __init__(self, serialization_dir):
        self._serialization_dir = serialization_dir

    def save_checkpoint(self, model, epoch, optimizer, global_step, save_best):
        if self._serialization_dir is not None:
            if save_best:
                model_path = os.path.join(self._serialization_dir, "model_state_best.th")
                model_state = model.state_dict()
                torch.save(model_state, model_path)

                training_state = {'epoch': epoch,
                                  'optimizer': optimizer.state_dict(),
                                  'global_step': global_step}
                training_path = os.path.join(self._serialization_dir, "training_state_best.th")
                torch.save(training_state, training_path)
            else:
                model_path = os.path.join(self._serialization_dir, "model_state_last.th")
                model_state = model.state_dict()
                torch.save(model_state, model_path)

                training_state = {'epoch': epoch,
                                  'optimizer': optimizer.state_dict(),
                                  'global_step': global_step}
                training_path = os.path.join(self._serialization_dir, "training_state_last.th")
                torch.save(training_state, training_path)

    @staticmethod
    def device_mapping(cuda_device: int):
        def inner_device_mapping(storage: torch.Storage):  # pylint: disable=unused-argument
            if cuda_device >= 0:
                return storage.cuda(cuda_device)
            else:
                return storage

        return inner_device_mapping

    @staticmethod
    def move_optimizer_to_cuda(optimizer):
        for param_group in optimizer.param_groups:
            for param in param_group['params']:
                if param.is_cuda:
                    param_state = optimizer.state[param]
                    for k in param_state.keys():
                        if isinstance(param_state[k], torch.Tensor):
                            param_state[k] = param_state[k].cuda(device=param.get_device())

    def restore_checkpoint(self, model, optimizer):

        training_state_path = os.path.join(self._serialization_dir, "training_state_last.th")
        model_path = os.path.join(self._serialization_dir, "model_state_last.th")

        if os.path.exists(training_state_path) and os.path.exists(model_path):
            model_state = torch.load(model_path, map_location='cuda:0')
            training_state = torch.load(training_state_path, map_location='cuda:0')
            model.load_state_dict(model_state)
            optimizer.load_state_dict(training_state["optimizer"])
            self.move_optimizer_to_cuda(optimizer)
            global_step = training_state["global_step"]
            if isinstance(training_state["epoch"], int):
                epoch_to_return = training_state["epoch"] + 1
            else:
                epoch_to_return = int(training_state["epoch"].split('.')[0]) + 1
            print("[i] Training resumed from last saved state.")
            return model, optimizer, epoch_to_return, global_step
        else:
            print("[i] Training started as new")
            return model, optimizer, 0, 0
