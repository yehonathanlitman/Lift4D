# Copyright (c) Meta Platforms, Inc. and affiliates.
import torch


class DepthModel:
    def __init__(self, model, device="cuda"):
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    def __call__(self, image):
        pass