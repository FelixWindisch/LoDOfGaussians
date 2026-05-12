#include "torch_interface.h"
#include "../hierarchy_loader.h"
#include "../hierarchy_writer.h"
#include "../traversal.h"
#include "../runtime_switching.h"
#include "../runtime_switching.h"
#include "../HierarchyCut.h"
#include "../morton.h"
#include "../utils.h"
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
LoadHierarchy(std::string filename)
{
	HierarchyLoader loader;
	
	std::vector<Eigen::Vector3f> pos;
	std::vector<SHs> shs;
	std::vector<float> alphas;
	std::vector<Eigen::Vector3f> scales;
	std::vector<Eigen::Vector4f> rot;
	std::vector<Node> nodes;
	std::vector<Box> boxes;
	
	loader.load(filename.c_str(), pos, shs, alphas, scales, rot, nodes, boxes);
	
	int P = pos.size();
	
	torch::TensorOptions options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
	torch::Tensor pos_tensor = torch::from_blob(pos.data(), {P, 3}, options).clone();
	torch::Tensor shs_tensor = torch::from_blob(shs.data(), {P, 16, 3}, options).clone();
	torch::Tensor alpha_tensor = torch::from_blob(alphas.data(), {P, 1}, options).clone();
	torch::Tensor scale_tensor = torch::from_blob(scales.data(), {P, 3}, options).clone();
	torch::Tensor rot_tensor = torch::from_blob(rot.data(), {P, 4}, options).clone();
	
	int N = nodes.size();
	torch::TensorOptions intoptions = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
	
	torch::Tensor nodes_tensor = torch::from_blob(nodes.data(), {N, 7}, intoptions).clone();
	torch::Tensor box_tensor = torch::from_blob(boxes.data(), {N, 2, 4}, options).clone();
	
	return std::make_tuple(pos_tensor, shs_tensor, alpha_tensor, scale_tensor, rot_tensor, nodes_tensor, box_tensor);
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
LoadDynamicHierarchy(std::string filename)
{
	HierarchyLoader loader;
	
	std::vector<Eigen::Vector3f> pos;
	std::vector<SHs> shs;
	std::vector<float> alphas;
	std::vector<Eigen::Vector3f> scales;
	std::vector<Eigen::Vector4f> rot;
	std::vector<HierarchyNode> nodes;
	int sh_size[] = {1,4,9, 16};
	int sh_degree;
	loader.loadDynamic(filename.c_str(), pos, shs, alphas, scales, rot, nodes, sh_degree);
	
	int P = pos.size();
	
	torch::TensorOptions options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
	torch::Tensor pos_tensor = torch::from_blob(pos.data(), {P, 3}, options).clone();
	
	torch::Tensor shs_tensor = torch::from_blob(shs.data(), {P, sh_size[sh_degree], 3}, options).clone();
	torch::Tensor alpha_tensor = torch::from_blob(alphas.data(), {P, 1}, options).clone();
	torch::Tensor scale_tensor = torch::from_blob(scales.data(), {P, 3}, options).clone();
	torch::Tensor rot_tensor = torch::from_blob(rot.data(), {P, 4}, options).clone();
	
	int N = nodes.size();
	printf("N: %d\n\n\n", N);
	torch::TensorOptions intoptions = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
	
	torch::Tensor nodes_tensor = torch::from_blob(nodes.data(), {N, 6}, intoptions).clone();
	
	return std::make_tuple(pos_tensor, shs_tensor, alpha_tensor, scale_tensor, rot_tensor, nodes_tensor);
}

void WriteHierarchy(
					std::string filename,
					torch::Tensor& pos,
					torch::Tensor& shs,
					torch::Tensor& opacities,
					torch::Tensor& log_scales,
					torch::Tensor& rotations,
					torch::Tensor& nodes,
					torch::Tensor& boxes)
{
	HierarchyWriter writer;
	
	int allP = pos.size(0);
	int allN = nodes.size(0);
	
	writer.write(
		filename.c_str(),
		allP,
		allN,
		(Eigen::Vector3f*)pos.cpu().contiguous().data_ptr<float>(),
		(SHs*)shs.cpu().contiguous().data_ptr<float>(),
		opacities.cpu().contiguous().data_ptr<float>(),
		(Eigen::Vector3f*)log_scales.cpu().contiguous().data_ptr<float>(),
		(Eigen::Vector4f*)rotations.cpu().contiguous().data_ptr<float>(),
		(Node*)nodes.cpu().contiguous().data_ptr<int>(),
		(Box*)boxes.cpu().contiguous().data_ptr<float>()
	);
}


void WriteDynamicHierarchy(
					std::string filename,
					torch::Tensor& pos,
					torch::Tensor& shs,
					torch::Tensor& opacities,
					torch::Tensor& log_scales,
					torch::Tensor& rotations,
					torch::Tensor& nodes,
					int SH_degree)
{
	HierarchyWriter writer;
	
	int allP = pos.size(0);
	int allN = nodes.size(0);
	
	
	writer.writeDynamic(
		filename.c_str(),
		allP,
		allN,
		(Eigen::Vector3f*)pos.cpu().contiguous().data_ptr<float>(),
		(SHs*)shs.cpu().contiguous().data_ptr<float>(),
		opacities.cpu().contiguous().data_ptr<float>(),
		(Eigen::Vector3f*)log_scales.cpu().contiguous().data_ptr<float>(),
		(Eigen::Vector4f*)rotations.cpu().contiguous().data_ptr<float>(),
		(HierarchyNode*)nodes.cpu().contiguous().data_ptr<int>(), SH_degree
	);
}

torch::Tensor
ExpandToTarget(torch::Tensor& nodes, int target)
{
	std::vector<int> indices = Traversal::expandToTarget((Node*)nodes.cpu().contiguous().data_ptr<int>(), target);
	torch::TensorOptions intoptions = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
	return torch::from_blob(indices.data(), {(int)indices.size()}, intoptions).clone();
}





int ExpandToSize(
torch::Tensor& nodes, 
torch::Tensor& boxes, 
float size, 
torch::Tensor& viewpoint, 
torch::Tensor& viewdir, 
torch::Tensor& render_indices,
torch::Tensor& parent_indices,
torch::Tensor& nodes_for_render_indices)
{
	return Switching::expandToSize(
	nodes.size(0), 
	size,
	nodes.contiguous().data_ptr<int>(), 
	boxes.contiguous().data_ptr<float>(),
	viewpoint.contiguous().data_ptr<float>(),
	viewdir.data_ptr<float>()[0], viewdir.data_ptr<float>()[1], viewdir.data_ptr<float>()[2],
	render_indices.contiguous().data_ptr<int>(),
	nullptr,
	parent_indices.contiguous().data_ptr<int>(),
	nodes_for_render_indices.contiguous().data_ptr<int>());
}

int ExpandToSizeDynamic(
torch::Tensor& nodes, 
torch::Tensor& positions,
torch::Tensor& scales, 
float size, 
torch::Tensor& viewpoint, 
torch::Tensor& viewdir, 
torch::Tensor& render_indices,
torch::Tensor& parent_indices,
torch::Tensor& nodes_for_render_indices)
{
	return Switching::expandToSizeDynamic(
	nodes.size(0), 
	size,
	nodes.contiguous().data_ptr<int>(), 
	positions.contiguous().data_ptr<float>(),
	scales.contiguous().data_ptr<float>(),
	viewpoint.contiguous().data_ptr<float>(),
	viewdir.data_ptr<float>()[0], viewdir.data_ptr<float>()[1], viewdir.data_ptr<float>()[2],
	render_indices.contiguous().data_ptr<int>(),
	nullptr,
	parent_indices.contiguous().data_ptr<int>(),
	nodes_for_render_indices.contiguous().data_ptr<int>());
}





void GetTsIndexed(
torch::Tensor& indices,
float size,
torch::Tensor& nodes,
torch::Tensor& boxes,
torch::Tensor& viewpoint, 
torch::Tensor& viewdir, 
torch::Tensor& ts,
torch::Tensor& num_kids)
{
	Switching::getTsIndexed(
	indices.size(0),
	indices.contiguous().data_ptr<int>(),
	size,
	nodes.contiguous().data_ptr<int>(), 
	boxes.contiguous().data_ptr<float>(),
	viewpoint.data_ptr<float>()[0], viewpoint.data_ptr<float>()[1], viewpoint.data_ptr<float>()[2],
	viewdir.data_ptr<float>()[0], viewdir.data_ptr<float>()[1], viewdir.data_ptr<float>()[2],
	ts.contiguous().data_ptr<float>(),
	num_kids.contiguous().data_ptr<int>(), 0);
}

void GetTsIndexedDynamic(
torch::Tensor& indices,
float size,
torch::Tensor& nodes,
torch::Tensor& positions,
torch::Tensor& scales,
torch::Tensor& viewpoint, 
torch::Tensor& viewdir, 
torch::Tensor& ts,
torch::Tensor& num_kids)
{
	Switching::getTsIndexedDynamic(
	indices.size(0),
	indices.contiguous().data_ptr<int>(),
	size,
	nodes.contiguous().data_ptr<int>(), 
	positions.contiguous().data_ptr<float>(),
	scales.contiguous().data_ptr<float>(),
	viewpoint.data_ptr<float>()[0], viewpoint.data_ptr<float>()[1], viewpoint.data_ptr<float>()[2],
	viewdir.data_ptr<float>()[0], viewdir.data_ptr<float>()[1], viewdir.data_ptr<float>()[2],
	ts.contiguous().data_ptr<float>(),
	num_kids.contiguous().data_ptr<int>(), 0);
}

void GetMortonCode(torch::Tensor& xyz, torch::Tensor& min, torch::Tensor& max, torch::Tensor& code)
{
	
	int P = xyz.sizes()[0];
	std::cout << P <<"WTF\n";
	MORTON::getMortonCode
	(
		xyz.contiguous().data_ptr<float>(),
		min.contiguous().data_ptr<float>(),
		max.contiguous().data_ptr<float>(),
		code.contiguous().data_ptr<int64_t>(), 
		P
	);
	
}


//int GetHierarchyCut(
//torch::Tensor& nodes, 
//torch::Tensor& positions,
//torch::Tensor& scales, 
//float size, 
//torch::Tensor& viewpoint, 
//torch::Tensor& viewdir, 
//torch::Tensor& render_indices)
//{
//	return HierarchyCut::getHierarchyCut(
//	nodes.size(0), 
//	size,
//	nodes.contiguous().data_ptr<int>(), 
//	positions.contiguous().data_ptr<float>(),
//	scales.contiguous().data_ptr<float>(),
//	viewpoint.contiguous().data_ptr<float>(),
//	viewdir.data_ptr<float>()[0], viewdir.data_ptr<float>()[1], viewdir.data_ptr<float>()[2],
//	render_indices.contiguous().data_ptr<int>(),
//	nullptr,
//	parent_indices.contiguous().data_ptr<int>(),
//	nodes_for_render_indices.contiguous().data_ptr<int>());
//}


std::tuple<torch::Tensor, torch::Tensor>
 GetSPTCut(
	int number_of_SPTs,
	torch::Tensor& gaussian_indices,
	torch::Tensor& SPT_starts,
	torch::Tensor& SPT_max,
	torch::Tensor& SPT_min,
	torch::Tensor& SPT_indices,
	torch::Tensor& SPT_distances)
{
	int* SPT_cut;
	int* SPT_counts;
	//int size = 5;
	//cudaMalloc(&d_result, size * sizeof(int));
	int size = Switching::getSPTCut(
		number_of_SPTs,
		gaussian_indices.contiguous().data_ptr<int>(),
		SPT_starts.contiguous().data_ptr<int>(),
		SPT_max.contiguous().data_ptr<float>(),
		SPT_min.contiguous().data_ptr<float>(),
		SPT_indices.contiguous().data_ptr<int>(),
		SPT_distances.contiguous().data_ptr<float>(),
		&SPT_cut,
		&SPT_counts);
	torch::Tensor SPT_counts_tensor = torch::from_blob(SPT_counts, {number_of_SPTs}, torch::kInt32).to(torch::kCUDA);
	torch::Tensor SPT_cut_tensor = torch::from_blob(SPT_cut, {size}, torch::kInt32).to(torch::kCUDA);
	cudaFree(SPT_counts);
	cudaFree(SPT_cut);
	return std::make_tuple(SPT_cut_tensor, SPT_counts_tensor);
}



std::tuple<torch::Tensor, torch::Tensor> ComputeRelocationCUDA(
	torch::Tensor& opacity_old,
	torch::Tensor& scale_old,
	torch::Tensor& N,
	torch::Tensor& binoms,
	const int n_max)
{
	const int P = opacity_old.size(0);
  
	torch::Tensor final_opacity = torch::full({P}, 0, opacity_old.options().dtype(torch::kFloat32));
	torch::Tensor final_scale = torch::full({3 * P}, 0, scale_old.options().dtype(torch::kFloat32));
	if(P != 0)
	{
		UTILS::ComputeRelocation(P,
			opacity_old.contiguous().data<float>(),
			scale_old.contiguous().data<float>(),
			N.contiguous().data<int>(),
			binoms.contiguous().data<float>(),
			n_max,
			final_opacity.contiguous().data<float>(),
			final_scale.contiguous().data<float>());
	}
	return std::make_tuple(final_opacity, final_scale);
}