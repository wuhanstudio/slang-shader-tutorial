# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numpy as np
import slangpy as spy

# CPU
c1 = 1.0 + 2.0j
c2 = 3.0 + 4.0j

c3 = c1 + c2
c4 = c1 - c2
c5 = c1 * c2
c6 = c1 / c2

print("CPU Complex Operations:")
print(f"c1 + c2 = {c3}")
print(f"c1 - c2 = {c4}")
print(f"c1 * c2 = {c5}")
print(f"c1 / c2 = {c6}")
print()

# GPU
device = spy.Device(
    type=spy.DeviceType.vulkan,
    enable_print=True
)

print(device)

program = device.load_program("complex.slang", ["test"])
kernel = device.create_compute_kernel(program)
kernel.dispatch(thread_count=[1, 1, 1])

device.flush_print()
