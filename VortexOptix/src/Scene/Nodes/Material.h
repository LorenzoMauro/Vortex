#pragma once
#include "Scene/Node.h"
#include "Shader.h"

namespace vtx::graph
{
	class Material : public Node {
	public:
		Material() : Node(NT_MATERIAL) {}

		void setShader(std::shared_ptr<Shader> _shader) {
			shader = _shader;
		}

		void traverse(const std::vector<std::shared_ptr<NodeVisitor>>& orderedVisitors) override;

		void accept(std::shared_ptr<NodeVisitor> visitor) override;

	public:
		std::shared_ptr<Shader> shader;
	};
}
