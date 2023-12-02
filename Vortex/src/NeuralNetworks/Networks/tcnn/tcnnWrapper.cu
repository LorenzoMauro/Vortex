#undef PI
#include <tiny-cuda-nn/config.h>

void testTCNN()
{
	// Configure the model
	nlohmann::json config = {
		{"loss", {
			{"otype", "L2"}
		}},
		{"optimizer", {
			{"otype", "Adam"},
			{"learning_rate", 1e-3},
		}},
		{"encoding", {
			{"otype", "HashGrid"},
			{"n_levels", 16},
			{"n_features_per_level", 2},
			{"log2_hashmap_size", 19},
			{"base_resolution", 16},
			{"per_level_scale", 2.0},
		}},
		{"network", {
			{"otype", "FullyFusedMLP"},
			{"activation", "ReLU"},
			{"output_activation", "None"},
			{"n_neurons", 64},
			{"n_hidden_layers", 2},
		}},
	};

	tcnn::TrainableModel model = tcnn::create_from_config(3, 3, config);

}
