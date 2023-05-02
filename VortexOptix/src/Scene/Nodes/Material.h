#pragma once
#include "Scene/Node.h"
#include "Shader/Shader.h"
#include "Scene/DataStructs/VertexAttribute.h"
#include "Scene/Utility/Operations.h"

namespace vtx::graph
{
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

	class Material : public Node {
	public:
		Material() : Node(NT_MATERIAL)
		{
			shader = ops::createNode<Shader>();
		}

		void setShader(std::shared_ptr<graph::Shader> _shader) {
			shader = _shader;
		}

		void init();

		std::shared_ptr<graph::Shader> getShader();

		size_t getArgumentBlockSize();

		char* getArgumentBlockData();

		void traverse(const std::vector<std::shared_ptr<NodeVisitor>>& orderedVisitors) override;

		void accept(std::shared_ptr<NodeVisitor> visitor) override;

	public:
		std::shared_ptr<graph::Shader>							shader;
		Handle<ITarget_argument_block>							argBlock;
		std::list<ParamInfo>									params;
		std::map<std::string, std::shared_ptr<EnumTypeInfo>>	mapEnumTypes;
		bool													isInitialized = false;
	};
}
