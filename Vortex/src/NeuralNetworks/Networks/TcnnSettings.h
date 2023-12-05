#pragma once
#include <map>
#include <string>
namespace vtx::network
{
    enum class TcnnEncodingType {
        Composite,
        Frequency,
        Grid,
        Identity,
        OneBlob,
        SphericalHarmonics,
        TriangleWave,
        TcnnEncodingTypeCount
    };


	inline static const char* TcnnEncodingTypeName[] = {
		"Composite",
		"Frequency",
		"Grid",
		"Identity",
		"OneBlob",
		"SphericalHarmonics",
		"TriangleWave"
	};

    inline static std::map<std::string, TcnnEncodingType> TcnnEncodingTypeNameToEnum =
    {
        { "Composite", TcnnEncodingType::Composite },
		{ "Frequency", TcnnEncodingType::Frequency },
		{ "Grid", TcnnEncodingType::Grid },
		{ "Identity", TcnnEncodingType::Identity },
		{ "OneBlob", TcnnEncodingType::OneBlob },
		{ "SphericalHarmonics", TcnnEncodingType::SphericalHarmonics },
		{ "TriangleWave", TcnnEncodingType::TriangleWave  }
    };

    enum class GridType {
        Hash,
        Tiled,
        Dense,
        GridTypeCount
    };

	inline static const char* GridTypeName[] = {
		"Hash",
		"Tiled",
		"Dense"
	};

	inline static std::map<std::string, GridType> GridTypeNameToEnum =
	{
		{"Hash", GridType::Hash},
		{"Tiled", GridType::Tiled},
		{"Dense", GridType::Dense}
	};

    enum class InterpolationType {
        Nearest,
        Linear,
        Smoothstep,
        InterpolationTypeCount
    };

    inline static const char* InterpolationTypeName[] = {
    	"Nearest",
    	"Linear",
    	"Smoothstep"
    };

    inline static std::map<std::string, InterpolationType> InterpolationTypeNameToEnum =
	{
		{"Nearest", InterpolationType::Nearest},
		{"Linear", InterpolationType::Linear},
		{"Smoothstep", InterpolationType::Smoothstep}
	};

    struct FrequencyEncoding {
        TcnnEncodingType otype = TcnnEncodingType::Frequency;
        int n_frequencies = 12;
    };

    struct GridEncoding {
        TcnnEncodingType  otype = TcnnEncodingType::Grid;
        GridType          type  = GridType::Hash;
        int               n_levels = 16;
        int               n_features_per_level = 2;
        int               log2_hashmap_size = 19;
        int               base_resolution = 16;
        float             per_level_scale = 2.0;
        InterpolationType interpolation = InterpolationType::Linear;
    };

    struct IdentityEncoding {
        TcnnEncodingType otype = TcnnEncodingType::Identity;
        float scale = 1.0;
        float offset = 1.0;
    };

    struct OneBlobEncoding {
        TcnnEncodingType otype = TcnnEncodingType::OneBlob;
        int n_bins = 16;
    };

    struct SphericalHarmonicsEncoding {
        TcnnEncodingType otype = TcnnEncodingType::SphericalHarmonics;
        int degree = 4;
    };

    struct TriangleWaveEncoding {
        TcnnEncodingType otype = TcnnEncodingType::TriangleWave;
        int n_frequencies = 12;
    };

    struct TcnnEncodingConfig
    {
        TcnnEncodingType otype = TcnnEncodingType::Identity;
        FrequencyEncoding frequencyEncoding;
        GridEncoding gridEncoding;
        IdentityEncoding identityEncoding;
        OneBlobEncoding oneBlobEncoding;
        SphericalHarmonicsEncoding sphericalHarmonicsEncoding;
        TriangleWaveEncoding triangleWaveEncoding;
    };

    struct TcnnCompositeEncodingConfig
    {
        TcnnEncodingConfig positionEncoding;
        TcnnEncodingConfig directionEncoding;
        TcnnEncodingConfig normalEncoding;
    };
}
