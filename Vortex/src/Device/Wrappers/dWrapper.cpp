#include "dWrapper.h"

#include "Device/CUDAChecks.h"

#undef min
#undef max

namespace vtx
{
    struct KernelStats {
        KernelStats(const char* description) : description(description) {}

        std::string description;
        int numLaunches = 0;
        float sumMS = 0, minMS = 0, maxMS = 0;
    };

    static std::map<std::string, KernelStats*> kernelStats;
    //static std::vector<KernelStats*> kernelStats;

    struct ProfilerEvent {
        ProfilerEvent() {
            CUDA_CHECK(cudaEventCreate(&start));
            CUDA_CHECK(cudaEventCreate(&stop));
        }

        void Sync() {
            VTX_ASSERT_CLOSE("ProfilerEvent Sync Fail", active);
            CUDA_CHECK(cudaEventSynchronize(start));
            CUDA_CHECK(cudaEventSynchronize(stop));

            float       ms  = 0;
			cudaError_t err = cudaEventElapsedTime(&ms, start, stop);
            if(err!=cudaSuccess)
            {
                VTX_INFO("Profiler Event Sync Fail on Kernle: {} Launches: {}", stats->description, stats->numLaunches);
                CUDA_CHECK_CONTINUE(err);
			}
            else
            {
                ++stats->numLaunches;
                if (stats->numLaunches == 1)
                    stats->sumMS = stats->minMS = stats->maxMS = ms;
                else {
                    stats->sumMS += ms;
                    stats->minMS = std::min(stats->minMS, ms);
                    stats->maxMS = std::max(stats->maxMS, ms);
                }
	            
            }

            active = false;
        }

        cudaEvent_t start, stop;
        bool active = false;
        KernelStats* stats = nullptr;
    };


    // Ring buffer
    static std::vector<ProfilerEvent> eventPool;
    static size_t eventPoolOffset = 0;
    static int maxPoolSize = 100;
    bool hasLooped = false;

    std::pair<cudaEvent_t, cudaEvent_t> vtx::GetProfilerEvents(const char* description) {
        if (eventPool.empty())
            eventPool.resize(maxPoolSize);  // how many? This is probably more than we need...

        if (eventPoolOffset == eventPool.size())
        {
	        hasLooped = true;
            eventPoolOffset = 0;
        }
		

        ProfilerEvent& pe = eventPool[eventPoolOffset++];
        if (pe.active)
        {
        	pe.Sync();
        }

        pe.active = true;
        pe.stats = nullptr;

        if(kernelStats.find(description) != kernelStats.end())
        {
	        pe.stats = kernelStats[description];
        }
        else
        {
            kernelStats.insert({ description, new KernelStats(description) });
	        pe.stats = kernelStats[description];
		}

        return { pe.start, pe.stop };
    }

    float vtx::GetKernelTimeMS(const char* description)
    {
        if(!hasLooped)
        {
            return 0.0f;
        }
        if (kernelStats.find(description) != kernelStats.end())
        {
            return kernelStats[description]->sumMS;
        }
        return 0.0f;
    }

    int vtx::GetKernelLaunches(const char* description)
    {
        if (!hasLooped)
        {
            return 0;
        }
	    if(kernelStats.find(description) != kernelStats.end())
	    {
	    		        return kernelStats[description]->numLaunches;
	    }
    }

    void vtx::resetKernelStats()
    {
        /*for(auto& [description, kS] : kernelStats)
        {
        	kS->numLaunches = 0;
	    	kS->sumMS = 0;
	    	kS->minMS = 0;
	    	kS->maxMS = 0;
        }*/

        kernelStats.clear();
        eventPoolOffset = 0;
        eventPool.clear();
        hasLooped = false;
    }
}

