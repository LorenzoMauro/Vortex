#pragma once
#ifndef SHADERFLAGS_H

// Defines for the DeviceShaderConfiguration::flags
#define IS_THIN_WALLED     (1u << 0)
// These flags are used to control which specific hit record is used.
#define USE_EMISSION       (1u << 1)
#define USE_CUTOUT_OPACITY (1u << 2)

#endif