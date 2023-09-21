// ======================================================================== //
// Copyright 2018 Ingo Wald                                                 //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#pragma once

namespace gdt {

	// Helper function for comparing floating point numbers
	template<typename T>
	inline __forceinline__ __both__ bool isEqual(const T& a, const T& b, const float maxDiff =0.00001f, const float maxRelDiff = FLT_EPSILON)
	{
		// Check if the numbers are really close -- needed
		// when comparing numbers near zero.
		const float diff = fabs(a - b);
		if (diff <= maxDiff)
			return true;

		const float asbA = fabs(a);
		const float asbB = fabs(b);

		if (const float largest = (asbB > asbA) ? asbB : asbA; diff <= largest * maxRelDiff)
			return true;
		return false;
	}

  // ------------------------------------------------------------------
  // ==
  // ------------------------------------------------------------------

	template<typename T>
	inline __forceinline__ __both__ bool operator==(const vec_t<T, 2>& a, const vec_t<T, 2>& b)
	{
		return (isEqual(a.x, b.x) && isEqual(a.y, b.y));
	}

	template<typename T>
	inline __forceinline__ __both__ bool operator==(const vec_t<T, 3>& a, const vec_t<T, 3>& b)
	{
		return (isEqual(a.x, b.x) && isEqual(a.y, b.y) && isEqual(a.z, b.z));
	}

	template<typename T>
	inline __forceinline__ __both__ bool operator==(const vec_t<T, 4>& a, const vec_t<T, 4>& b)
	{
		return (isEqual(a.x, b.x) && isEqual(a.y, b.y) && isEqual(a.z, b.z) && isEqual(a.w, b.w));
	}
  
  // ------------------------------------------------------------------
  // !=
  // ------------------------------------------------------------------
	template<typename T>
	inline __forceinline__ __both__ bool operator!=(const vec_t<T, 2>& a, const vec_t<T, 2>& b)
	{
		return (!isEqual(a.x, b.x) || !isEqual(a.y, b.y));
	}

	template<typename T>
	inline __forceinline__ __both__ bool operator!=(const vec_t<T, 3>& a, const vec_t<T, 3>& b)
	{
		return (!isEqual(a.x, b.x) || !isEqual(a.y, b.y) || !isEqual(a.z, b.z));
	}

	template<typename T>
	inline __forceinline__ __both__ bool operator!=(const vec_t<T, 4>& a, const vec_t<T, 4>& b)
	{
		return (!isEqual(a.x, b.x) || !isEqual(a.y, b.y) || !isEqual(a.z, b.z) || !isEqual(a.w, b.w));
	}

} // ::gdt
