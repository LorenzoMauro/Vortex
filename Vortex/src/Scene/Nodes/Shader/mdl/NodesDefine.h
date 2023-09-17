#pragma once

////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////// VORTEX PRINCIPLED SHADER NODES  ///////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
#define VORTEX_PRINCIPLED_MODULE "VortexPrincipledShader.mdl"
#define VORTEX_PRINCIPLED_FUNCTION "principledVortex"

#define THIN_WALLED_SOCKET "isThinWalled"

#define ALBEDO_SOCKET "BaseColor"
#define METALLIC_SOCKET "Metallic"
#define SPECULAR_TINT_SOCKET "SpecularTint"
#define IOR_SOCKET "Ior"
#define ROUGHNESS_SOCKET "Roughness"
#define ANISOTROPY_SOCKET "Anisotropy"
#define ANISOTROPY_ROTATION_SOCKET "Anisotropy_rotation"
#define NORMALMAP_SOCKET "Normal"

#define TRANSMISSION_SOCKET "Transmission"
#define TRANSMISSION_ROUGHNESS_SOCKET "TransmissionRoughness"

#define COAT_AMOUNT_SOCKET "Coat"
#define COAT_ROUGHNESS_SOCKET "CoatRoughness"
#define COAT_TRANSPARENCY_SOCKET "CoatTransparency"
#define COAT_COLOR_SOCKET "CoatColor"
#define COAT_NORMALMAP_SOCKET "ClearcoatNormal"

#define FILM_AMOUNT_SOCKET "FilmAmmount"
#define FILM_IOR_SOCKET "FilmIor"
#define FILM_THICKNESS_SOCKET "FilmThickness"

#define EMISSION_INTENSITY_SOCKET "EmissionIntensity"
#define EMISSION_COLOR_SOCKET "EmissionColor"

#define SHEEN_AMOUNT_SOCKET "SheenAmmount"
#define SHEEN_TINT_SOCKET "SheenTint"
#define SHEEN_ROUGHNESS_SOCKET "SheenRoghness"


////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////// VORTEX CUSTOM FUNCTIONS  //////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////

#define VORTEX_FUNCTIONS_MODULE "VortexFunctions.mdl"

#define VF_COLOR_TEXTURE "ColorTexture"
#define VF_COLOR_TEXTURE_TEXTURE_SOCKET "texture"
#define VF_COLOR_TEXTURE_COORDINATES_SOCKET "coordinate"

#define VF_MONO_TEXTURE "MonoTexture"
#define VF_MONO_TEXTURE_TEXTURE_SOCKET "texture"
#define VF_MONO_TEXTURE_COORDINATES_SOCKET "coordinate"

#define VF_BUMP_TEXTURE "BumpTexture"
#define VF_BUMP_TEXTURE_TEXTURE_SOCKET "texture"
#define VF_BUMP_TEXTURE_FACTOR_SOCKET "factor"
#define VF_BUMP_TEXTURE_COORDINATES_SOCKET "coordinate"

#define VF_NORMAL_TEXTURE "NormalTexture"
#define VF_NORMAL_TEXTURE_TEXTURE_SOCKET "texture"
#define VF_NORMAL_TEXTURE_FACTOR_SOCKET "factor"
#define VF_NORMAL_TEXTURE_COORDINATES_SOCKET "coordinate"
#define VF_NORMAL_TEXTURE_FLIP_U_SOCKET "flip_tangent_u"
#define VF_NORMAL_TEXTURE_FLIP_V_SOCKET "flip_tangent_v"

#define VF_TEXTURE_TRANSFORM "transformCoordinate"
#define VF_TEXTURE_TRANSFORM_TRANSLATION_SOCKET "translation"
#define VF_TEXTURE_TRANSFORM_ROTATION_SOCKET "rotation"
#define VF_TEXTURE_TRANSFORM_SCALING_SOCKET "scaling"

#define VF_MIX_NORMAL "add_detail_normal"
#define VF_MIX_NORMAL_BASE_SOCKET "n"
#define VF_MIX_NORMAL_LAYER_SOCKET "nd"

#define VF_GET_COLOR_CHANNEL				"getChannel"
#define VF_GET_COLOR_CHANNEL_COLOR_SOCKET	"inputColor"
#define VF_GET_COLOR_CHANNEL_CHANNEL_SOCKET "channel"
