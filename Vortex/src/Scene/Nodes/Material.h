#pragma once
#include "Device/OptixWrapper.h"
#include "Scene/Node.h"
#include "Scene/Utility/Operations.h"

#include "MDL/MdlWrapper.h"

namespace mi
{
	namespace neuraylib
	{
		class ITarget_code;
	}
}

namespace vtx::graph
{
	namespace shader
	{
		class ShaderNode;
	}

	// Possible enum values if any.
	struct EnumValue
	{
		std::string name;
		int         value;

		EnumValue(std::string name, const int value)
			: name(std::move(name))
			, value(value)
		{
		}
	};

	// Info for an enum type.
	struct EnumTypeInfo
	{
		std::vector<EnumValue> values;

		// Adds a enum value and its integer value to the enum type info.
		void add(const std::string& name, int value)
		{
			values.emplace_back(name, value);
		}
	};

	// Material parameter information structure.
	class ParamInfo
	{
	public:
		enum ParamKind
		{
			PK_UNKNOWN,
			PK_FLOAT,
			PK_FLOAT2,
			PK_FLOAT3,
			PK_COLOR,
			PK_ARRAY,
			PK_BOOL,
			PK_INT,
			PK_ENUM,
			PK_STRING,
			PK_TEXTURE,
			PK_LIGHT_PROFILE,
			PK_BSDF_MEASUREMENT
		};

		ParamInfo() :
			m_index(-1),
			m_kind(PK_UNKNOWN),
			m_arrayElemKind(PK_UNKNOWN),
			m_arraySize(-1),
			m_arrayPitch(-1),
			m_dataPtr(nullptr),
			m_rangeMin(0.0f),
			m_rangeMax(1.0f),
			m_enumInfo(nullptr)
		{
		}

		ParamInfo(const size_t index,
		          char const* name,
		          char const* displayName,
		          char const* groupName,
		          const ParamKind kind,
		          const ParamKind arrayElemKind,
		          const mi::Size   arraySize,
		          const mi::Size   arrayPitch,
		          char* dataPtr,
		          const EnumTypeInfo* enumInfo = nullptr)
			: m_index(index)
			, m_name(name)
			, m_displayName(displayName)
			, m_groupName(groupName)
			, m_kind(kind)
			, m_arrayElemKind(arrayElemKind)
			, m_arraySize(arraySize)
			, m_arrayPitch(arrayPitch)
			, m_dataPtr(dataPtr)
			, m_rangeMin(0.0f)
			, m_rangeMax(1.0f)
			, m_enumInfo(enumInfo)
		{
		}

		// Get data as T&.
		template<typename T>
		T& data()
		{
			return *reinterpret_cast<T*>(m_dataPtr);
		}

		// Get data as const T&.
		template<typename T>
		const T& data() const
		{
			return *reinterpret_cast<const T*>(m_dataPtr);
		}

		const char* displayName() const
		{
			return m_displayName.c_str();
		}
		void setDisplayName(const char* displayName)
		{
			m_displayName = displayName;
		}

		const char* groupName() const
		{
			return m_groupName.c_str();
		}
		void setGroupName(const char* groupName)
		{
			m_groupName = groupName;
		}

		ParamKind kind() const
		{
			return m_kind;
		}

		ParamKind arrayElemKind() const
		{
			return m_arrayElemKind;
		}
		mi::Size arraySize() const
		{
			return m_arraySize;
		}
		mi::Size arrayPitch() const
		{
			return m_arrayPitch;
		}

		float& rangeMin()
		{
			return m_rangeMin;
		}
		float rangeMin() const
		{
			return m_rangeMin;
		}
		float& rangeMax()
		{
			return m_rangeMax;
		}
		float rangeMax() const
		{
			return m_rangeMax;
		}

		template<typename T, int N = 1>
		void updateRange()
		{
			T* valPtr = &data<T>();
			for (int i = 0; i < N; ++i)
			{
				const float val = static_cast<float>(valPtr[i]);
				if (val < m_rangeMin)
					m_rangeMin = val;
				if (m_rangeMax < val)
					m_rangeMax = val;
			}
		}

		const EnumTypeInfo* enumInfo() const
		{
			return m_enumInfo;
		}

		std::string getName()
		{
			return m_name;
		}

	private:
		size_t					m_index;
		std::string				m_name;
		std::string				m_displayName;
		std::string				m_groupName;
		ParamKind				m_kind;
		ParamKind				m_arrayElemKind;
		mi::Size				m_arraySize;
		mi::Size				m_arrayPitch;   // the distance between two array elements
		char*					m_dataPtr;
		float					m_rangeMin;
		float					m_rangeMax;
		const EnumTypeInfo*		m_enumInfo;
	};

	struct Configuration {
		// The state of the expressions :
		bool isThinWalledConstant = false;
		bool isThinWalled = false;
		bool isSurfaceBsdfValid = false;
		bool isBackfaceBsdfValid = false;
		bool isSurfaceEdfValid = false;
		bool isSurfaceIntensityConstant = true;
		bool isSurfaceIntensityModeConstant = true;
		bool isBackfaceEdfValid = false;
		bool isBackfaceIntensityConstant = true;
		bool isBackfaceIntensityModeConstant = true;
		bool useBackfaceEdf = false;
		bool useBackfaceIntensity = false;
		bool useBackfaceIntensityMode = false;
		bool isIorConstant = true;
		bool isVdfValid = false;
		bool isAbsorptionCoefficientConstant = true;
		bool useVolumeAbsorption = false;
		bool isScatteringCoefficientConstant = true;
		bool isDirectionalBiasConstant = true;
		bool useVolumeScattering = false;
		bool isCutoutOpacityConstant = false;
		bool useCutoutOpacity = false;
		bool isHairBsdfValid = false;

		// The constant expression values:
		math::vec3f		surfaceIntensity{ 0.0f };
		int				surfaceIntensityMode{ 0 };
		math::vec3f		backfaceIntensity{ 0.0f };
		int				backfaceIntensityMode{ 0 };
		math::vec3f		ior{ 1.0f };
		math::vec3f		absorptionCoefficient{ 0.0f };
		math::vec3f		scatteringCoefficient{ 0.0f };
		float			directionalBias{ 0.0f };
		float			cutoutOpacity{ 1.0f };

		// These parameters are added to manage PBR shaders
		bool* emissivityToggle = nullptr;
		bool* opacityToggle = nullptr;
	};

	struct FunctionNames {

		FunctionNames() = default;

		FunctionNames(const std::string& suffix) {
			init = "__direct_callable__init_" + suffix;
			thinWalled = "__direct_callable__thin_walled_" + suffix;

			surfaceScattering = "__direct_callable__surface_scattering_" + suffix;

			surfaceEmissionEmission = "__direct_callable__surface_emission_emission_" + suffix;
			surfaceEmissionIntensity = "__direct_callable__surface_emission_intensity_" + suffix;
			surfaceEmissionMode = "__direct_callable__surface_emission_mode_" + suffix;

			backfaceScattering = "__direct_callable__backface_scattering_" + suffix;

			backfaceEmissionEmission = "__direct_callable__backface_emission_emission_" + suffix;
			backfaceEmissionIntensity = "__direct_callable__backface_emission_intensity_" + suffix;
			backfaceEmissionMode = "__direct_callable__backface_emission_mode_" + suffix;

			ior = "__direct_callable__ior_" + suffix;

			volumeAbsorptionCoefficient = "__direct_callable__volume_absorption_coefficient_" + suffix;
			volumeScatteringCoefficient = "__direct_callable__volume_scattering_coefficient_" + suffix;
			volumeDirectionalBias = "__direct_callable__volume_directional_bias_" + suffix;

			geometryCutoutOpacity = "__direct_callable__geometry_cutout_opacity_" + suffix;

			hairBsdf = "__direct_callable__hair_" + suffix;
		}

		std::string init;
		std::string thinWalled;
		std::string surfaceScattering;
		std::string surfaceEmissionEmission;
		std::string surfaceEmissionIntensity;
		std::string surfaceEmissionMode;
		std::string backfaceScattering;
		std::string backfaceEmissionEmission;
		std::string backfaceEmissionIntensity;
		std::string backfaceEmissionMode;
		std::string ior;
		std::string volumeAbsorptionCoefficient;
		std::string volumeScatteringCoefficient;
		std::string volumeDirectionalBias;
		std::string geometryCutoutOpacity;
		std::string hairBsdf;
	};

	struct DevicePrograms
	{
		std::shared_ptr<optix::ProgramOptix> pgInit;
		std::shared_ptr<optix::ProgramOptix> pgEvaluateMaterial;
		std::shared_ptr<optix::ProgramOptix> pgThinWalled;

		std::shared_ptr<optix::ProgramOptix> pgSurfaceScatteringSample;
		std::shared_ptr<optix::ProgramOptix> pgSurfaceScatteringEval;
		std::shared_ptr<optix::ProgramOptix> pgSurfaceScatteringAuxiliary;

		std::shared_ptr<optix::ProgramOptix> pgBackfaceScatteringSample;
		std::shared_ptr<optix::ProgramOptix> pgBackfaceScatteringEval;
		std::shared_ptr<optix::ProgramOptix> pgBackfaceScatteringAuxiliary;

		std::shared_ptr<optix::ProgramOptix> pgSurfaceEmissionEval;
		std::shared_ptr<optix::ProgramOptix> pgSurfaceEmissionIntensity;
		std::shared_ptr<optix::ProgramOptix> pgSurfaceEmissionIntensityMode;

		std::shared_ptr<optix::ProgramOptix> pgBackfaceEmissionEval;
		std::shared_ptr<optix::ProgramOptix> pgBackfaceEmissionIntensity;
		std::shared_ptr<optix::ProgramOptix> pgBackfaceEmissionIntensityMode;

		std::shared_ptr<optix::ProgramOptix> pgIor;

		std::shared_ptr<optix::ProgramOptix> pgVolumeAbsorptionCoefficient;
		std::shared_ptr<optix::ProgramOptix> pgVolumeScatteringCoefficient;
		std::shared_ptr<optix::ProgramOptix> pgVolumeDirectionalBias;

		std::shared_ptr<optix::ProgramOptix> pgGeometryCutoutOpacity;

		std::shared_ptr<optix::ProgramOptix> pgHairSample;
		std::shared_ptr<optix::ProgramOptix> pgHairEval;

		//bool isEmissive = false;
		//bool isThinWalled = true;
		//bool hasOpacity = false;
		//unsigned int  flags;
	};

	class Material : public Node {
	public:
		Material() : Node(NT_MATERIAL)
		{
		}

		void init();

		size_t getArgumentBlockSize();

		char* getArgumentBlockData();

		void traverse(const std::vector<std::shared_ptr<NodeVisitor>>& orderedVisitors) override;

		//void accept(std::shared_ptr<NodeVisitor> visitor) override;

		std::shared_ptr<graph::shader::ShaderNode>              materialGraph;
		mi::base::Handle<mi::neuraylib::ITarget_argument_block> argBlock;
		std::map<std::string, std::vector<graph::ParamInfo>>    params;
		std::map<std::string, std::shared_ptr<EnumTypeInfo>>    mapEnumTypes;
		bool                                                    isInitialized = false;
		bool                                                    useAsLight    = false;


		//////////////////////////////////////////////////////////////////////////////////////////
		///////////////////////// RELOCATED FROM OLD SHADER NODE /////////////////////////////////
		//////////////////////////////////////////////////////////////////////////////////////////

		void createPrograms();
		void createChildResources();
		const Configuration& getConfiguration();
		const DevicePrograms& getPrograms();

		std::vector<std::shared_ptr<graph::Texture>>         getTextures();
		std::vector<std::shared_ptr<graph::BsdfMeasurement>> getBsdfs();
		std::vector<std::shared_ptr<graph::LightProfile>>    getLightProfiles();

		bool											useEmission();
		bool											useOpacity();
		bool                                            isThinWalled();

		std::string                                                         name;			// example : "bsdf_diffuse_reflection"
		std::string                                                         materialDbName;	// example : "bsdf_diffuse_reflection"
		std::string                                                         path;			// example : "mdl\\bsdf_diffuse_reflection.mdl"
		std::string															materialCallName;

		Configuration                                       config;
		mi::base::Handle<mi::neuraylib::ITarget_code const> targetCode;
		DevicePrograms                                      devicePrograms;

		//Child Resources
		std::vector<std::shared_ptr<graph::Texture>>                        textures;
		std::vector<std::shared_ptr<graph::LightProfile>>					lightProfiles;
		std::vector<std::shared_ptr<graph::BsdfMeasurement>>                bsdfMeasurements;

		mi::base::Handle<mi::neuraylib::ITarget_value_layout const>   argLayout;
		mi::base::Handle<mi::neuraylib::ITarget_argument_block const> argumentBlock;
	};
}
