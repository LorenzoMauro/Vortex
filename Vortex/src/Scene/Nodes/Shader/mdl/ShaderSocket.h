#include "Scene/Node.h"

namespace vtx::graph::shader
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



	struct Annotation
	{
		std::string              displayName;
		std::string              description;
		std::string              groupName;
		float range[2] = { 0.0f, 1.0f };
		bool isValid = false;

		std::string print()
		{
			std::string result = "Annotation: \n";
			result += "Display Name: " + displayName + "\n";
			result += "Description: " + description + "\n";
			result += "Group Name: " + groupName + "\n";
			result += "Range: " + std::to_string(range[0]) + " - " + std::to_string(range[1]) + "\n";
			return result;
		}

	};

	// Material parameter information structure.
	class ParameterInfo
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

		ParameterInfo() :
			index(-1),
			kind(PK_UNKNOWN),
			arrayElemKind(PK_UNKNOWN),
			arraySize(-1),
			arrayPitch(-1),
			dataPtr(nullptr),
			enumInfo(nullptr),
			expressionKind(mi::neuraylib::IType::Kind::TK_FORCE_32_BIT)
		{
		}

		// Get data as T&.
		template<typename T>
		T& data()
		{
			return *reinterpret_cast<T*>(dataPtr);
		}

		// Get data as const T&.
		template<typename T>
		const T& data() const
		{
			return *reinterpret_cast<const T*>(dataPtr);
		}

		template<typename T, int N = 1>
		void updateRange()
		{
			T* valPtr = &data<T>();
			for (int i = 0; i < N; ++i)
			{
				const float val = static_cast<float>(valPtr[i]);
				if (val < annotation.range[0])
					annotation.range[0]= val;
				if (annotation.range[1] < val)
					annotation.range[1] = val;
			}
		}

		size_t                     index;
		std::string                argumentName;
		ParamKind                  kind;
		ParamKind                  arrayElemKind;
		mi::Size                   arraySize;
		mi::Size                   arrayPitch; // the distance between two array elements
		char*                      dataPtr;
		const EnumTypeInfo*        enumInfo;
		mi::neuraylib::IType::Kind expressionKind;
		Annotation                 annotation;
	};

	struct ShaderNodeSocket
	{
		std::shared_ptr<ShaderNode>                  node;
		ParameterInfo                                parameterInfo;
		vtxID                                        Id;
		mi::base::Handle<mi::neuraylib::IExpression> directExpression;
		vtxID                                        linkId;
	};

	using ShaderInputSockets = std::map<std::string, ShaderNodeSocket>;
}