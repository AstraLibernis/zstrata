# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 AstraLibernis
#
# Source this before building or running: `. tools/env.sh`
# Points the Vulkan loader, the RADV driver, the validation layer and the shader tools at the
# copy persisted under ~/.local/vulkan (the container's /usr is wiped on every restart).

VKPREFIX="${VKPREFIX:-$HOME/.local/vulkan/usr}"
export PATH="$VKPREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$VKPREFIX/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export VK_DRIVER_FILES="$VKPREFIX/share/vulkan/icd.d/radeon_icd.x86_64.json"
export VK_ADD_LAYER_PATH="$VKPREFIX/share/vulkan/explicit_layer.d"
# RADV prints a "not a conformant Vulkan implementation" banner on every start; it is noise
# here.
# libdrm looks for amdgpu.ids (PCI ids -> GPU names) under /usr, which this container does not
# keep; the copy lives with the rest of the stack. Without it libdrm writes "No such file" to
# stderr, which makes `zig build test` fail its GPU tests with no message.
export AMDGPU_ASIC_ID_TABLE_PATHS="$VKPREFIX/share/libdrm"
export MESA_VK_IGNORE_CONFORMANCE_WARNING=1
