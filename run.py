#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import sys
import os
import torch
import torch_npu
import torchair
from torch_npu.testing.testcase import TestCase, run_tests
torch.ops.load_library("libcustom_ops.so")


class TestCustomAdd(TestCase):
    def test_cema_lr_ops_ops(self):
        length = [8, 2048]
        x = torch.rand(length, device='cpu', dtype=torch.float)
        y = torch.rand(length, device='cpu', dtype=torch.float)
        output = torch.ops.ascendc_ops.ascendc_add(x.npu(), y.npu()).cpu()
        cpuout = torch.add(x, y)
        self.assertRtolEqual(output, cpuout)


if __name__ == "__main__":
    run_tests()
