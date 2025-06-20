# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import slangpy as spy

device = spy.Device(
    type=spy.DeviceType.vulkan,
    enable_print=True
)

print(device)

program = device.load_program("print.slang", ["main"])
kernel = device.create_compute_kernel(program)
kernel.dispatch(thread_count=[2, 2, 1])

device.flush_print()
