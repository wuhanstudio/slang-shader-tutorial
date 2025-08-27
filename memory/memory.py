# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numpy as np
import slangpy as spy

device = spy.Device(
    type=spy.DeviceType.vulkan,
    enable_print=True
)

print(device)

array = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]).reshape((2, 2, 3)).astype(np.float32)
array_cpu = array.flatten()

print("===== CPU =====")
for i in range(len(array_cpu)):
    print(f"array[{i}] = {array_cpu[i]}")

program = device.load_program("memory.slang", ["main"])
kernel = device.create_compute_kernel(program)

# Input buffers
buffer_a = device.create_buffer(
    element_count=len(array_cpu),
    struct_type=kernel.reflection.main.a,
    usage=spy.BufferUsage.shader_resource,
    data=array,
)

kernel.dispatch(thread_count=[10, 1, 1], a=buffer_a)

print("===== GPU =====")
device.flush_print()
