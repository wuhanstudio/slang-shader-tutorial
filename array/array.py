# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numpy as np
import slangpy as spy

device = spy.Device(
    type=spy.DeviceType.vulkan,
    enable_print=True
)

print(device)

program = device.load_program("array.slang", ["main"])
kernel = device.create_compute_kernel(program)

# Input buffers
buffer_a = device.create_buffer(
    element_count=1024,
    struct_type=kernel.reflection.main.a,
    usage=spy.BufferUsage.shader_resource,
    data=np.linspace(0, 1, 1024, dtype=np.float32),
)

buffer_b = device.create_buffer(
    element_count=1024,
    struct_type=kernel.reflection.main.b,
    usage=spy.BufferUsage.shader_resource,
    data=np.linspace(1, 0, 1024, dtype=np.float32),
)

# Output buffer
buffer_c = device.create_buffer(
    element_count=1024,
    struct_type=kernel.reflection.main.c,
    usage=spy.BufferUsage.unordered_access,
)

kernel.dispatch(thread_count=[1024, 1, 1], a=buffer_a, b=buffer_b, c=buffer_c)

# Print the output
data = buffer_c.to_numpy().view(np.float32)

print(data.shape)
print(data)
