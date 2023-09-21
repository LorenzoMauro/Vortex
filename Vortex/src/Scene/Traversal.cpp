#include "Traversal.h"
#include "Graph.h"

namespace vtx
{
	
	void NodeVisitor::visitBegin(const std::shared_ptr<graph::Node>& node)
	{
		if (collectWidthsAndDepths)
		{
			bool isRoot = false;
			if(depthStack.empty())
			{
				depthStack.push(0);
				depthWidth.resize(1);
				isRoot = true;
			}
			else
			{
				depthStack.push(depthStack.top() + 1);
			}
			
			// Replace the value for the node's ID with the current depth and width

			int depth = depthStack.top();
			int width = widthStack.empty()? 0 : widthStack.top();

			if (depth == depthWidth.size())
			{
				depthWidth.resize(depth + 1);
				depthWidth[depth] = 0;
			}
			else if (!isRoot)
			{
				depthWidth[depth] += 1;
			}
			//VTX_INFO("Node ID: {} Name: {} \t\t Depth: {} Width: {} Overall Width: {}", node->getID(), node->name, depth, width, overallWidth);
			nodesDepthsAndWidths[node->getID()] = { depth, width, depthWidth[depth] };// , currentWidth + parent
			if (!widthStack.empty())
			{
				int& currentWidth = widthStack.top();
				currentWidth += 1;
			}

			widthStack.push(0);

			nodesParents[node->getID()] = parentPath;
			parentPath.push_back(node->getID());
		}

		if (collectTransforms)
		{
			bool foundTransform = false;
			for (const std::shared_ptr<graph::Node> child : node->getChildren())
			{
				if (std::shared_ptr<graph::Transform> transform = child->as<graph::Transform>(); transform != nullptr)
				{
					tmpTransforms.push(currentTransform);
					foundTransform = true;
					currentTransform = currentTransform * transform->affineTransform;
					transform->globalTransform = currentTransform;
					break;
				}
			}
			resetTransforms.push(foundTransform);
		}
	}
	void NodeVisitor::visitEnd(const std::shared_ptr<graph::Node>& node)
	{
		if (collectWidthsAndDepths)
		{
			//VTX_INFO("Node ID: {} Name: {} Popping: \t\t Depth Stack Size: {} Width Stack Size: {}", node->getID(), node->name, depthStack.size(), widthStack.size());
			depthStack.pop();
			widthStack.pop();
			parentPath.pop_back();
		}
		if (collectTransforms)
		{
			if (resetTransforms.top())
			{
				currentTransform = tmpTransforms.top();
				tmpTransforms.pop();

			}
			resetTransforms.pop();
		}
	}

	void NodeVisitor::visit(const std::shared_ptr<graph::Transform>& transform)
	{
		visit(std::dynamic_pointer_cast<graph::Node>(transform));
	}
	void NodeVisitor::visit(const std::shared_ptr<graph::Instance>& instance)
	{
		visit(std::dynamic_pointer_cast<graph::Node>(instance));
	}
	void NodeVisitor::visit(const std::shared_ptr<graph::Group>& group)
	{
		visit(std::dynamic_pointer_cast<graph::Node>(group));
	}
	void NodeVisitor::visit(const std::shared_ptr<graph::Mesh>& mesh)
	{
		visit(std::dynamic_pointer_cast<graph::Node>(mesh));
	}
	void NodeVisitor::visit(const std::shared_ptr<graph::Material>& material)
	{
		visit(std::dynamic_pointer_cast<graph::Node>(material));
	}
	void NodeVisitor::visit(const std::shared_ptr<graph::Camera>& camera)
	{
		visit(std::dynamic_pointer_cast<graph::Node>(camera));
	}
	void NodeVisitor::visit(const std::shared_ptr<graph::Renderer>& renderer)
	{
		visit(std::dynamic_pointer_cast<graph::Node>(renderer));
	}
	void NodeVisitor::visit(const std::shared_ptr<graph::Texture>& texture)
	{
		visit(std::dynamic_pointer_cast<graph::Node>(texture));
	}
	void NodeVisitor::visit(const std::shared_ptr<graph::BsdfMeasurement>& bsdfMeasurement)
	{
		visit(std::dynamic_pointer_cast<graph::Node>(bsdfMeasurement));
	}
	void NodeVisitor::visit(const std::shared_ptr<graph::LightProfile>& lightProfile)
	{
		visit(std::dynamic_pointer_cast<graph::Node>(lightProfile));
	}
	void NodeVisitor::visit(const std::shared_ptr<graph::EnvironmentLight>& node)
	{
		visit(std::dynamic_pointer_cast<graph::Node>(node));
	}
	void NodeVisitor::visit(const std::shared_ptr<graph::MeshLight>& node)
	{
		visit(std::dynamic_pointer_cast<graph::Node>(node));
	}

	void NodeVisitor::visit(const std::shared_ptr<graph::shader::ShaderNode>& shaderNode)
	{
		visit(std::dynamic_pointer_cast<graph::Node>(shaderNode));
	}
	void NodeVisitor::visit(const std::shared_ptr<graph::shader::DiffuseReflection>& shaderNode)
	{
		visit(std::dynamic_pointer_cast<graph::shader::ShaderNode>(shaderNode));
	}
	void NodeVisitor::visit(const std::shared_ptr<graph::shader::MaterialSurface>& shaderNode)
	{
		visit(std::dynamic_pointer_cast<graph::shader::ShaderNode>(shaderNode));
	}
	void NodeVisitor::visit(const std::shared_ptr<graph::shader::Material>& shaderNode)
	{
		visit(std::dynamic_pointer_cast<graph::shader::ShaderNode>(shaderNode));
	}
	void NodeVisitor::visit(const std::shared_ptr<graph::shader::ImportedNode>& shaderNode)
	{
		visit(std::dynamic_pointer_cast<graph::shader::ShaderNode>(shaderNode));
	}
	void NodeVisitor::visit(const std::shared_ptr<graph::shader::PrincipledMaterial>& shaderNode)
	{
		visit(std::dynamic_pointer_cast<graph::shader::ShaderNode>(shaderNode));
	}
	void NodeVisitor::visit(const std::shared_ptr<graph::shader::ColorTexture>& shaderNode)
	{
		visit(std::dynamic_pointer_cast<graph::shader::ShaderNode>(shaderNode));
	}
	void NodeVisitor::visit(const std::shared_ptr<graph::shader::MonoTexture>& shaderNode)
	{
		visit(std::dynamic_pointer_cast<graph::shader::ShaderNode>(shaderNode));
	}
	void NodeVisitor::visit(const std::shared_ptr<graph::shader::NormalTexture>& shaderNode)
	{
		visit(std::dynamic_pointer_cast<graph::shader::ShaderNode>(shaderNode));
	}
	void NodeVisitor::visit(const std::shared_ptr<graph::shader::BumpTexture>& shaderNode)
	{
		visit(std::dynamic_pointer_cast<graph::shader::ShaderNode>(shaderNode));
	}
	void NodeVisitor::visit(const std::shared_ptr<graph::shader::TextureTransform>& shaderNode)
	{
		visit(std::dynamic_pointer_cast<graph::shader::ShaderNode>(shaderNode));
	}
	void NodeVisitor::visit(const std::shared_ptr<graph::shader::NormalMix>& shaderNode)
	{
		visit(std::dynamic_pointer_cast<graph::shader::ShaderNode>(shaderNode));
	}
	void NodeVisitor::visit(const std::shared_ptr<graph::shader::GetChannel>& shaderNode)
	{
		visit(std::dynamic_pointer_cast<graph::shader::ShaderNode>(shaderNode));
	}
}
