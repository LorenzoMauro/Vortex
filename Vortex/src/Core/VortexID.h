#pragma once

#ifndef VTXIDX_H
#define VTXIDX_H
#include <cstdint>
#include <limits>

constexpr unsigned int INVALID_INDEX = std::numeric_limits<unsigned int>::quiet_NaN();

namespace vtx {
	typedef uint32_t vtxID ;
}
#endif
