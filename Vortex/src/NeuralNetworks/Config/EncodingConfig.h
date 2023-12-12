#pragma once
#include <map>
#include <string>

namespace vtx::network::config
{
    enum class EncodingType {
        Composite,
        Frequency,
        Grid,
        Identity,
        OneBlob,
        SphericalHarmonics,
        TriangleWave,
        EncodingTypeCount
    };


    inline static const char* EncodingTypeName[] = {
        "Composite",
        "Frequency",
        "Grid",
        "Identity",
        "OneBlob",
        "SphericalHarmonics",
        "TriangleWave"
    };

    inline static std::map<std::string, EncodingType> EncodingTypeNameToEnum =
    {
        { "Composite", EncodingType::Composite },
        { "Frequency", EncodingType::Frequency },
        { "Grid", EncodingType::Grid },
        { "Identity", EncodingType::Identity },
        { "OneBlob", EncodingType::OneBlob },
        { "SphericalHarmonics", EncodingType::SphericalHarmonics },
        { "TriangleWave", EncodingType::TriangleWave  }
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
        EncodingType otype = EncodingType::Frequency;
        int n_frequencies = 12;
        bool operator==(const FrequencyEncoding& other) const
        {
	        return n_frequencies == other.n_frequencies;
        }
    };

    struct GridEncoding {
        EncodingType  otype = EncodingType::Grid;
        GridType          type = GridType::Hash;
        int               n_levels = 16;
        int               n_features_per_level = 2;
        int               log2_hashmap_size = 19;
        int               base_resolution = 16;
        float             per_level_scale = 2.0;
        InterpolationType interpolation = InterpolationType::Linear;

        bool operator==(const GridEncoding& other) const
        {
	        return type == other.type &&
				n_levels == other.n_levels &&
				n_features_per_level == other.n_features_per_level &&
				log2_hashmap_size == other.log2_hashmap_size &&
				base_resolution == other.base_resolution &&
				per_level_scale == other.per_level_scale &&
				interpolation == other.interpolation;
		}
    };

    struct IdentityEncoding {
        EncodingType otype = EncodingType::Identity;
        float scale = 1.0;
        float offset = 1.0;
        bool operator==(const IdentityEncoding& other) const
        {
	        return scale == other.scale && offset == other.offset;
        }
    };

    struct OneBlobEncoding {
        EncodingType otype = EncodingType::OneBlob;
        int n_bins = 16;
        bool operator==(const OneBlobEncoding& other) const
        {
	        return n_bins == other.n_bins;
		}
    };

    struct SphericalHarmonicsEncoding {
        EncodingType otype = EncodingType::SphericalHarmonics;
        int degree = 4;
        bool operator==(const SphericalHarmonicsEncoding& other) const
        {
	        return degree == other.degree;
		}
    };

    struct TriangleWaveEncoding {
        EncodingType otype = EncodingType::TriangleWave;
        int n_frequencies = 12;
        bool operator==(const TriangleWaveEncoding& other) const
        {
	        return n_frequencies == other.n_frequencies;
        }
    };

    struct EncodingConfig
    {
        bool operator==(const EncodingConfig& other) const
        {
	        	return otype == other.otype &&
				frequencyEncoding.n_frequencies == other.frequencyEncoding.n_frequencies &&
				gridEncoding.type == other.gridEncoding.type &&
				gridEncoding.n_levels == other.gridEncoding.n_levels &&
				gridEncoding.n_features_per_level == other.gridEncoding.n_features_per_level &&
				gridEncoding.log2_hashmap_size == other.gridEncoding.log2_hashmap_size &&
				gridEncoding.base_resolution == other.gridEncoding.base_resolution &&
				gridEncoding.per_level_scale == other.gridEncoding.per_level_scale &&
				gridEncoding.interpolation == other.gridEncoding.interpolation &&
				identityEncoding.scale == other.identityEncoding.scale &&
				identityEncoding.offset == other.identityEncoding.offset &&
				oneBlobEncoding.n_bins == other.oneBlobEncoding.n_bins &&
				sphericalHarmonicsEncoding.degree == other.sphericalHarmonicsEncoding.degree &&
				triangleWaveEncoding.n_frequencies == other.triangleWaveEncoding.n_frequencies;
		}

        EncodingType otype = EncodingType::Identity;
        FrequencyEncoding frequencyEncoding;
        GridEncoding gridEncoding;
        IdentityEncoding identityEncoding;
        OneBlobEncoding oneBlobEncoding;
        SphericalHarmonicsEncoding sphericalHarmonicsEncoding;
        TriangleWaveEncoding triangleWaveEncoding;
    };

}
