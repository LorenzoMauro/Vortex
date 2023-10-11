#pragma once
#include <map>

#include "CUDABuffer.h"
#include "Core/VortexID.h"

namespace vtx
{
#define VAR_NAME(x) #x

	template<typename T, typename B>
	struct DeviceData
	{
		CUDABuffer imageBuffer;
		B          resourceBuffers;
		bool       isDirty;

		const T& getHostImage()
		{
			return hostImage;
		}

		T& editableHostImage()
		{
			isDirty = true;
			return hostImage;
		}

		T* getDeviceImage()
		{
			if (isDirty)
			{
				T* devicePtr = imageBuffer.upload(hostImage);
				isDirty      = false;
				return devicePtr;
			}
			return imageBuffer.template castedPointer<T>();
		}

		CUdeviceptr getDevicePtr()
		{
			return (CUdeviceptr)getDeviceImage();
		}

		~DeviceData()
		{
			imageBuffer.free();
		}

		void freeResourceBuffer()
		{
			resourceBuffers.~B();
			resourceBuffers = B();
		}

	private:
		T          hostImage;
	};

	template<typename T, typename B>
	class DeviceDataMap
	{
	public:

		bool contains(const vtxID& id) const
		{
			return deviceDataMap.find(id) != deviceDataMap.end();
		}

		DeviceData<T,B>& operator[](vtxID id)
		{
			checkContains(id);
			return deviceDataMap[id];
		}

		DeviceData<T,B>& getAndCreate(vtxID id)
		{
			checkContains(id);
			return deviceDataMap.at(id);
		}

		void insert(vtxID id, const T& hostImg)
		{
			deviceDataMap[id].editableHostImage()= hostImg;
			isMapChanged = true;
		}

		B& getResourceBuffers(vtxID id)
		{
			return deviceDataMap[id].resourceBuffers;
		}

		void erase(vtxID id)
		{
			//checkContains(id);
			deviceDataMap.erase(id);
			isMapChanged = true;
		}

		bool isMapChanged = false;

		int size() const
		{
			return deviceDataMap.size();
		}

		class Iterator
		{
		public:
			Iterator(typename std::map<vtxID, DeviceData<T,B>>::iterator it) : itr(it) {}

			Iterator& operator++()
			{
				++itr;
				return *this;
			}

			bool operator!=(const Iterator& other) const
			{
				return itr != other.itr;
			}

			const vtxID& operator*() const
			{
				return itr->first;
			}

		private:
			typename std::map<vtxID, DeviceData<T,B>>::iterator itr;
		};

		Iterator begin()
		{
			return Iterator(deviceDataMap.begin());
		}

		Iterator end()
		{
			return Iterator(deviceDataMap.end());
		}

	private:
		void checkContains(const vtxID& id) const
		{
			if (!contains(id))
			{
				VTX_WARN("The requested Node Id either doesn't exist or has not been registered!");
				throw std::runtime_error("Invalid vtxID provided to DeviceDataMap.");
			}
		}

		std::map<vtxID, DeviceData<T, B>> deviceDataMap;
	};
}
