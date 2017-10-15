/**
 * @file      rasterize.cu
 * @brief     CUDA-accelerated rasterization pipeline.
 * @authors   Skeleton code: Yining Karl Li, Kai Ninomiya, Shuai Shao (Shrek)
 * @date      2012-2016
 * @copyright University of Pennsylvania & STUDENT
 */

#include <time.h>
#include <random>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/random.h>
#include <util/checkCUDAError.h>
#include <util/tiny_gltf_loader.h>
#include "rasterizeTools.h"
#include "rasterize.h"
#include <glm/gtc/random.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>

#define BILINEAR_INTERP 0
#define BLINN 1
#define SSAO 0
#define SSAA 2 
#define TOON 0
#define BLOOM 0
#define TILE_BASED_RENDER 1
#define BACKFACE_CULLING 1
#define SHADING 0 // 0: Solid, 1: Wireframe, 2: Point
#define DEMO 0

namespace {

	typedef unsigned short VertexIndex;
	typedef glm::vec3 VertexAttributePosition;
	typedef glm::vec3 VertexAttributeNormal;
	typedef glm::vec2 VertexAttributeTexcoord;
	typedef unsigned char TextureData;

	typedef unsigned char BufferByte;

	enum PrimitiveType{
		Point = 1,
		Line = 2,
		Triangle = 3
	};

	struct VertexOut {
		glm::vec4 pos;

		// TODO: add new attributes to your VertexOut
		// The attributes listed below might be useful, 
		// but always feel free to modify on your own

		 glm::vec3 eyePos;	// eye space position used for shading
		 glm::vec3 eyeNor;	// eye space normal used for shading, cuz normal will go wrong after perspective transformation
		 glm::vec3 col;
		 glm::vec2 texcoord0;
		 TextureData* dev_diffuseTex = NULL;
		 int texWidth, texHeight;
		// ...
	};

	struct Primitive {
		PrimitiveType primitiveType = Triangle;	// C++ 11 init
		TextureData* tex = NULL;
		VertexOut v[3];
		int size[2];
	};

	struct Fragment {
		glm::vec3 color;
		float z;

		// TODO: add new attributes to your Fragment
		// The attributes listed below might be useful, 
		// but always feel free to modify on your own

		 glm::vec3 eyePos;	// eye space position used for shading
		 glm::vec3 eyeNor;
		 VertexAttributeTexcoord texcoord0;
		 TextureData* dev_diffuseTex = NULL;
		 int size[2];
		 
	};

	struct PrimitiveDevBufPointers {
		int primitiveMode;	//from tinygltfloader macro
		PrimitiveType primitiveType;
		int numPrimitives;
		int numIndices;
		int numVertices;

		// Vertex In, const after loaded
		VertexIndex* dev_indices;
		VertexAttributePosition* dev_position;
		VertexAttributeNormal* dev_normal;
		VertexAttributeTexcoord* dev_texcoord0;

		// Materials, add more attributes when needed
		TextureData* dev_diffuseTex;
		int diffuseTexWidth;
		int diffuseTexHeight;
		// TextureData* dev_specularTex;
		// TextureData* dev_normalTex;
		// ...

		// Vertex Out, vertex used for rasterization, this is changing every frame
		VertexOut* dev_verticesOut;

		// TODO: add more attributes when needed
	};

}

static std::map<std::string, std::vector<PrimitiveDevBufPointers>> mesh2PrimitivesMap;


static int width = 0;
static int height = 0;
static int ow = 0;
static int oh = 0;
static int tilesize = 16;
static int numTilesW, numTilesH, numTiles;

static int totalNumPrimitives = 0;
static Primitive *dev_primitives = NULL;
static Fragment *dev_fragmentBuffer = NULL;
static glm::vec3 *dev_framebuffer = NULL;
static glm::vec2 *dev_noise = NULL;

static int *dev_count = NULL;
static int *dev_tile = NULL;
static float * dev_depth = NULL;	// you might need this buffer when doing depth test


__constant__ int mx[5][5] = {
	{ 1, 4, 7, 4, 1 },
	{ 4,16,26,16,4 },
	{ 7,26,41,26,7 },
	{ 4,16,26,16,4 },
	{ 1,4,7,4,1 }
};

/**
 * Kernel that writes the image to the OpenGL PBO directly.
 */
__global__ 
void sendImageToPBO(uchar4 *pbo, int w, int h, glm::vec3 *image) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * w);

    if (x < w && y < h) {
        glm::vec3 color;
		#if SSAA
			int it = int(SSAA);
			for (int i = 0; i < it; i++)
			for (int j = 0; j < it; j++)
			{
				int sx = it * x;
				int sy = it * y;
				int sw = it * w;
				color.x += glm::clamp(image[sx + i + (sy + j) * sw].x, 0.0f, 1.0f) * 255.0;
				color.y += glm::clamp(image[sx + i + (sy + j) * sw].y, 0.0f, 1.0f) * 255.0;
				color.z += glm::clamp(image[sx + i + (sy + j) * sw].z, 0.0f, 1.0f) * 255.0;
			}
			color /= float(1 << it);
		#else
			color.x = glm::clamp(image[index].x, 0.0f, 1.0f) * 255.0;
			color.y = glm::clamp(image[index].y, 0.0f, 1.0f) * 255.0;
			color.z = glm::clamp(image[index].z, 0.0f, 1.0f) * 255.0;
		#endif
        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

#define COL(C) (C / 255.0)
#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))
#define MAX(X, Y) (((X) > (Y)) ? (X) : (Y))


__host__ __device__
int clamp(int v, int a, int b)
{
	return MIN(MAX(a, v), b);
}

__host__ __device__
glm::vec3 getTexColor(TextureData* tex, int stride, int u, int v)
{
	int idx = (u + v * stride) * 3;
	return glm::vec3(COL(tex[idx + 0]),
		COL(tex[idx + 1]),
		COL(tex[idx + 2]));
}

template<class T>
__host__ __device__
T lerp(float v, T a, T b)
{
	return a * (1.0f - v) + v * b;
}

__host__ __device__
glm::vec2 getRandom(glm::vec2 *noise, int u, int v, int randomSize, int screenSize)
{
	int x = MIN(randomSize * ((float)u / screenSize), 7);
	int y = MIN(randomSize * ((float)v / screenSize), 7);
	glm::vec2 r = noise[x + y * randomSize];
	return glm::normalize(r * 2.0f - glm::vec2(1.0f));
}

__host__ __device__
float AO(Fragment *fb, int w, int h, glm::vec2 coord, glm::vec3 *buffer, int ww, int hh, glm::vec2 tcoord, glm::vec2 uv, glm::vec3 p, glm::vec3 cnorm)
{
	int gx = coord.x + uv.x;
	int gy = coord.y + uv.y;
	int x = tcoord.x + uv.x;
	int y = tcoord.y + uv.y;
	float scale = 1.0f;
	float bias = 0.05f;
	float ao_a = 5.0f;
	if (gx < w && gy < h && gx >=0 && gy >= 0) {
		int index;
		glm::vec3 diff;
		if (x < ww && y < hh && x >= 0 && y >= 0) {
			index = x + (y * ww);
			diff = buffer[index] - p;
		} else {
			index = gx + (gy * w);
			diff = fb[index].eyePos - p;
		}
		float d = glm::length(diff) * scale;
		if (d == 0) return 0;
		glm::vec3 v = glm::normalize(diff);
		float r = MAX(0.0f, glm::dot(cnorm, v) - bias) * (1.0f / (1.0f + d)) * ao_a;
		return r;
	}
	return 0;
}

__global__
void render(int w, int h, Fragment *fragmentBuffer, glm::vec3 *framebuffer, glm::vec2 *noise) {
	extern __shared__ glm::vec3 buffer[];

    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * w);
	
	// Cache position to shared memory
	buffer[threadIdx.x + blockDim.x * threadIdx.y] = fragmentBuffer[index].eyePos;
	__syncthreads();

	if (x < w && y < h) {

		auto fb = fragmentBuffer[index];
		auto out = glm::vec3(0);
		auto col = fb.color;
		auto nor = fb.eyeNor;
		auto pos = buffer[threadIdx.x + blockDim.x * threadIdx.y];

		glm::vec3 lightPos = glm::vec3(60, 60, 60);
		glm::vec3 lightDir = glm::normalize(lightPos - pos);
		float ambient = 0.2f;
		float shininess = 32.0f;
		float diffuse = 0;
		float specular = 0;

		// Blinn Shading Model
		// https://en.wikipedia.org/wiki/Blinn%E2%80%93Phong_shading_model

		#if BLINN
			glm::vec3 viewDir = glm::normalize(-pos);
			glm::vec3 halfDir = glm::normalize(lightDir + viewDir);
			float specAngle = glm::max(glm::dot(halfDir, nor), 0.0f);
			specular = glm::pow(specAngle, shininess);
			diffuse = glm::max(0.0f, glm::dot(nor, lightDir));
		#else
			diffuse = glm::max(0.0f, glm::dot(nor, lightDir));
		#endif
		
		if (fb.dev_diffuseTex != NULL) {
			float texWidth = fb.size[0];
			float texHeight = fb.size[1];
			auto tex = fb.dev_diffuseTex;
			auto texcoord = fb.texcoord0;

#if !BILINEAR_INTERP
			int u = texcoord.x * texWidth;
			int v = texcoord.y * texHeight;
			out = getTexColor(tex, texWidth, u, v);
#else
			// Bilinear
			float fx = texcoord.x * texWidth;
			float fy = texcoord.y * texHeight;
			int cx = clamp((int)fx, 0, texWidth - 1);
			int cy = clamp((int)fy, 0, texHeight - 1);
			float dx = fx - cx;
			float dy = fy - cy;
			auto x0y0 = getTexColor(tex, texWidth, cx + 0, cy + 0);
			auto x1y0 = getTexColor(tex, texWidth, cx + 1, cy + 0);
			auto x0y1 = getTexColor(tex, texWidth, cx + 0, cy + 1);
			auto x1y1 = getTexColor(tex, texWidth, cx + 1, cy + 1);
			out = lerp<glm::vec3>(dy, lerp<glm::vec3>(dx, x0y0, x1y0), lerp<glm::vec3>(dx, x0y1, x1y1));
#endif
		}
		else {
			out = col;
		}

// Toon Shader
#if TOON
		ambient = 0.1f;
		specular = specular > 0.75f ? 1.0f : 0.0f;
		if (diffuse < 0.1f) diffuse = 0.0f;
		else if (diffuse < 0.5f) diffuse = 0.5f;
		else diffuse = 0.8f;
		out = out * diffuse + glm::vec3(1, 1, 1) * specular + out * ambient + glm::vec3(1, 1, 1) * specular;
#else 
		out = out * diffuse + glm::vec3(1, 1, 1) * specular + out * ambient + glm::vec3(1, 1, 1) * specular;
#endif

		

// Default SSAO: 4 samples per fragment (4 iterations)
#if SSAO
		// SSAO
		// https://www.gamedev.net/articles/programming/graphics/a-simple-and-practical-approach-to-ssao-r2753/

		float z = -glm::abs((float)fb.z) / glm::abs(INT_MIN);
		int iterations = 4;
		float ao = 0.0f;
		glm::vec2 v[4] = { glm::vec2(1,0), glm::vec2(-1,0), glm::vec2(0,1), glm::vec2(0,-1) };
		glm::vec2 rand = getRandom(noise, x, y, 8, w);
		glm::vec2 xy(x, y);

		#pragma unroll
			for (int i = 0; i < iterations; i++)
			{
				if (z < 0.00000001f) continue;
				float sampleR = clamp(1.f / z, 0.0f, 10.0f) * 3.0f;
			
				glm::vec2 coord1 = glm::reflect(v[i], rand) * sampleR;
				glm::vec2 coord2 = glm::vec2(coord1.x * 0.707f - coord1.y * 0.707f, coord1.x * 0.707f + coord1.y * 0.707f);
				ao += AO(fragmentBuffer, w, h, xy, buffer, blockDim.x, blockDim.y, glm::vec2(threadIdx.x, threadIdx.y), coord1 * 0.25f, pos, nor);
				ao += AO(fragmentBuffer, w, h, xy, buffer, blockDim.x, blockDim.y, glm::vec2(threadIdx.x, threadIdx.y), coord2 * 0.50f, pos, nor);
				ao += AO(fragmentBuffer, w, h, xy, buffer, blockDim.x, blockDim.y, glm::vec2(threadIdx.x, threadIdx.y), coord1 * 0.75f, pos, nor);
				ao += AO(fragmentBuffer, w, h, xy, buffer, blockDim.x, blockDim.y, glm::vec2(threadIdx.x, threadIdx.y), coord2 * 1.00f, pos, nor);
			}
		
		ao /= (float)iterations * 4.0f;
		out *= (1 - ao);
#endif
		framebuffer[index] = out;
    }
}

/**
 * Called once at the beginning of the program to allocate memory.
 */
void rasterizeInit(int w, int h) {
	ow = w;
	oh = h;
	#if SSAA
		width = w * SSAA;
		height = h * SSAA;
	#else
		width = w;
		height = h;
	#endif

		numTilesW = width / tilesize;
		numTilesH = height / tilesize;
		numTiles = numTilesW * numTilesH;

	cudaFree(dev_fragmentBuffer);
	cudaMalloc(&dev_fragmentBuffer, width * height * sizeof(Fragment));
	cudaMemset(dev_fragmentBuffer, 0, width * height * sizeof(Fragment));
    cudaFree(dev_framebuffer);
    cudaMalloc(&dev_framebuffer,   width * height * sizeof(glm::vec3));
    cudaMemset(dev_framebuffer, 0, width * height * sizeof(glm::vec3));
    
	cudaFree(dev_depth);
	cudaMalloc(&dev_depth, width * height * sizeof(float));

	cudaFree(dev_count);
	cudaMalloc(&dev_count, numTiles * sizeof(int));

	cudaFree(dev_tile);

	checkCUDAError("rasterizeInit");
}

__global__
void initDepth(int w, int h, float * depth)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < w && y < h)
	{
		int index = x + (y * w);
		depth[index] = float(INT_MAX);
	}
}


/**
* kern function with support for stride to sometimes replace cudaMemcpy
* One thread is responsible for copying one component
*/
__global__ 
void _deviceBufferCopy(int N, BufferByte* dev_dst, const BufferByte* dev_src, int n, int byteStride, int byteOffset, int componentTypeByteSize) {
	
	// Attribute (vec3 position)
	// component (3 * float)
	// byte (4 * byte)

	// id of component
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (i < N) {
		int count = i / n;
		int offset = i - count * n;	// which component of the attribute

		for (int j = 0; j < componentTypeByteSize; j++) {
			
			dev_dst[count * componentTypeByteSize * n 
				+ offset * componentTypeByteSize 
				+ j]

				= 

			dev_src[byteOffset 
				+ count * (byteStride == 0 ? componentTypeByteSize * n : byteStride) 
				+ offset * componentTypeByteSize 
				+ j];
		}
	}
	

}

__global__
void _nodeMatrixTransform(
	int numVertices,
	VertexAttributePosition* position,
	VertexAttributeNormal* normal,
	glm::mat4 MV, glm::mat3 MV_normal) {

	// vertex id
	int vid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (vid < numVertices) {
		position[vid] = glm::vec3(MV * glm::vec4(position[vid], 1.0f));
		normal[vid] = glm::normalize(MV_normal * normal[vid]);
	}
}

glm::mat4 getMatrixFromNodeMatrixVector(const tinygltf::Node & n) {
	
	glm::mat4 curMatrix(1.0);

	const std::vector<double> &m = n.matrix;
	if (m.size() > 0) {
		// matrix, copy it

		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				curMatrix[i][j] = (float)m.at(4 * i + j);
			}
		}
	} else {
		// no matrix, use rotation, scale, translation

		if (n.translation.size() > 0) {
			curMatrix[3][0] = n.translation[0];
			curMatrix[3][1] = n.translation[1];
			curMatrix[3][2] = n.translation[2];
		}

		if (n.rotation.size() > 0) {
			glm::mat4 R;
			glm::quat q;
			q[0] = n.rotation[0];
			q[1] = n.rotation[1];
			q[2] = n.rotation[2];

			R = glm::mat4_cast(q);
			curMatrix = curMatrix * R;
		}

		if (n.scale.size() > 0) {
			curMatrix = curMatrix * glm::scale(glm::vec3(n.scale[0], n.scale[1], n.scale[2]));
		}
	}

	return curMatrix;
}

void traverseNode (
	std::map<std::string, glm::mat4> & n2m,
	const tinygltf::Scene & scene,
	const std::string & nodeString,
	const glm::mat4 & parentMatrix
	) 
{
	const tinygltf::Node & n = scene.nodes.at(nodeString);
	glm::mat4 M = parentMatrix * getMatrixFromNodeMatrixVector(n);
	n2m.insert(std::pair<std::string, glm::mat4>(nodeString, M));

	auto it = n.children.begin();
	auto itEnd = n.children.end();

	for (; it != itEnd; ++it) {
		traverseNode(n2m, scene, *it, M);
	}
}

void rasterizeSetBuffers(const tinygltf::Scene & scene) {

	totalNumPrimitives = 0;

	std::map<std::string, BufferByte*> bufferViewDevPointers;

	// 1. copy all `bufferViews` to device memory
	{
		std::map<std::string, tinygltf::BufferView>::const_iterator it(
			scene.bufferViews.begin());
		std::map<std::string, tinygltf::BufferView>::const_iterator itEnd(
			scene.bufferViews.end());

		for (; it != itEnd; it++) {
			const std::string key = it->first;
			const tinygltf::BufferView &bufferView = it->second;
			if (bufferView.target == 0) {
				continue; // Unsupported bufferView.
			}

			const tinygltf::Buffer &buffer = scene.buffers.at(bufferView.buffer);

			BufferByte* dev_bufferView;
			cudaMalloc(&dev_bufferView, bufferView.byteLength);
			cudaMemcpy(dev_bufferView, &buffer.data.front() + bufferView.byteOffset, bufferView.byteLength, cudaMemcpyHostToDevice);

			checkCUDAError("Set BufferView Device Mem");

			bufferViewDevPointers.insert(std::make_pair(key, dev_bufferView));

		}
	}



	// 2. for each mesh: 
	//		for each primitive: 
	//			build device buffer of indices, materail, and each attributes
	//			and store these pointers in a map
	{

		std::map<std::string, glm::mat4> nodeString2Matrix;
		auto rootNodeNamesList = scene.scenes.at(scene.defaultScene);

		{
			auto it = rootNodeNamesList.begin();
			auto itEnd = rootNodeNamesList.end();
			for (; it != itEnd; ++it) {
				traverseNode(nodeString2Matrix, scene, *it, glm::mat4(1.0f));
			}
		}


		// parse through node to access mesh

		auto itNode = nodeString2Matrix.begin();
		auto itEndNode = nodeString2Matrix.end();
		for (; itNode != itEndNode; ++itNode) {

			const tinygltf::Node & N = scene.nodes.at(itNode->first);
			const glm::mat4 & matrix = itNode->second;
			const glm::mat3 & matrixNormal = glm::transpose(glm::inverse(glm::mat3(matrix)));

			auto itMeshName = N.meshes.begin();
			auto itEndMeshName = N.meshes.end();

			for (; itMeshName != itEndMeshName; ++itMeshName) {

				const tinygltf::Mesh & mesh = scene.meshes.at(*itMeshName);

				auto res = mesh2PrimitivesMap.insert(std::pair<std::string, std::vector<PrimitiveDevBufPointers>>(mesh.name, std::vector<PrimitiveDevBufPointers>()));
				std::vector<PrimitiveDevBufPointers> & primitiveVector = (res.first)->second;

				// for each primitive
				for (size_t i = 0; i < mesh.primitives.size(); i++) {
					const tinygltf::Primitive &primitive = mesh.primitives[i];

					if (primitive.indices.empty())
						return;

					// TODO: add new attributes for your PrimitiveDevBufPointers when you add new attributes
					VertexIndex* dev_indices = NULL;
					VertexAttributePosition* dev_position = NULL;
					VertexAttributeNormal* dev_normal = NULL;
					VertexAttributeTexcoord* dev_texcoord0 = NULL;

					// ----------Indices-------------

					const tinygltf::Accessor &indexAccessor = scene.accessors.at(primitive.indices);
					const tinygltf::BufferView &bufferView = scene.bufferViews.at(indexAccessor.bufferView);
					BufferByte* dev_bufferView = bufferViewDevPointers.at(indexAccessor.bufferView);

					// assume type is SCALAR for indices
					int n = 1;
					int numIndices = indexAccessor.count;
					int componentTypeByteSize = sizeof(VertexIndex);
					int byteLength = numIndices * n * componentTypeByteSize;

					dim3 numThreadsPerBlock(128);
					dim3 numBlocks((numIndices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
					cudaMalloc(&dev_indices, byteLength);
					_deviceBufferCopy << <numBlocks, numThreadsPerBlock >> > (
						numIndices,
						(BufferByte*)dev_indices,
						dev_bufferView,
						n,
						indexAccessor.byteStride,
						indexAccessor.byteOffset,
						componentTypeByteSize);


					checkCUDAError("Set Index Buffer");


					// ---------Primitive Info-------

					// Warning: LINE_STRIP is not supported in tinygltfloader
					int numPrimitives;
					PrimitiveType primitiveType;
					switch (primitive.mode) {
					case TINYGLTF_MODE_TRIANGLES:
						primitiveType = PrimitiveType::Triangle;
						numPrimitives = numIndices / 3;
						break;
					case TINYGLTF_MODE_TRIANGLE_STRIP:
						primitiveType = PrimitiveType::Triangle;
						numPrimitives = numIndices - 2;
						break;
					case TINYGLTF_MODE_TRIANGLE_FAN:
						primitiveType = PrimitiveType::Triangle;
						numPrimitives = numIndices - 2;
						break;
					case TINYGLTF_MODE_LINE:
						primitiveType = PrimitiveType::Line;
						numPrimitives = numIndices / 2;
						break;
					case TINYGLTF_MODE_LINE_LOOP:
						primitiveType = PrimitiveType::Line;
						numPrimitives = numIndices + 1;
						break;
					case TINYGLTF_MODE_POINTS:
						primitiveType = PrimitiveType::Point;
						numPrimitives = numIndices;
						break;
					default:
						// output error
						break;
					};


					// ----------Attributes-------------

					auto it(primitive.attributes.begin());
					auto itEnd(primitive.attributes.end());

					int numVertices = 0;
					// for each attribute
					for (; it != itEnd; it++) {
						const tinygltf::Accessor &accessor = scene.accessors.at(it->second);
						const tinygltf::BufferView &bufferView = scene.bufferViews.at(accessor.bufferView);

						int n = 1;
						if (accessor.type == TINYGLTF_TYPE_SCALAR) {
							n = 1;
						}
						else if (accessor.type == TINYGLTF_TYPE_VEC2) {
							n = 2;
						}
						else if (accessor.type == TINYGLTF_TYPE_VEC3) {
							n = 3;
						}
						else if (accessor.type == TINYGLTF_TYPE_VEC4) {
							n = 4;
						}

						BufferByte * dev_bufferView = bufferViewDevPointers.at(accessor.bufferView);
						BufferByte ** dev_attribute = NULL;

						numVertices = accessor.count;
						int componentTypeByteSize;

						// Note: since the type of our attribute array (dev_position) is static (float32)
						// We assume the glTF model attribute type are 5126(FLOAT) here

						if (it->first.compare("POSITION") == 0) {
							componentTypeByteSize = sizeof(VertexAttributePosition) / n;
							dev_attribute = (BufferByte**)&dev_position;
						}
						else if (it->first.compare("NORMAL") == 0) {
							componentTypeByteSize = sizeof(VertexAttributeNormal) / n;
							dev_attribute = (BufferByte**)&dev_normal;
						}
						else if (it->first.compare("TEXCOORD_0") == 0) {
							componentTypeByteSize = sizeof(VertexAttributeTexcoord) / n;
							dev_attribute = (BufferByte**)&dev_texcoord0;
						}

						std::cout << accessor.bufferView << "  -  " << it->second << "  -  " << it->first << '\n';

						dim3 numThreadsPerBlock(128);
						dim3 numBlocks((n * numVertices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
						int byteLength = numVertices * n * componentTypeByteSize;
						cudaMalloc(dev_attribute, byteLength);

						_deviceBufferCopy << <numBlocks, numThreadsPerBlock >> > (
							n * numVertices,
							*dev_attribute,
							dev_bufferView,
							n,
							accessor.byteStride,
							accessor.byteOffset,
							componentTypeByteSize);

						std::string msg = "Set Attribute Buffer: " + it->first;
						checkCUDAError(msg.c_str());
					}

					// malloc for VertexOut
					VertexOut* dev_vertexOut;
					cudaMalloc(&dev_vertexOut, numVertices * sizeof(VertexOut));
					checkCUDAError("Malloc VertexOut Buffer");

					// ----------Materials-------------

					// You can only worry about this part once you started to 
					// implement textures for your rasterizer
					TextureData* dev_diffuseTex = NULL;
					int diffuseTexWidth = 0;
					int diffuseTexHeight = 0;
					if (!primitive.material.empty()) {
						const tinygltf::Material &mat = scene.materials.at(primitive.material);
						printf("material.name = %s\n", mat.name.c_str());

						if (mat.values.find("diffuse") != mat.values.end()) {
							std::string diffuseTexName = mat.values.at("diffuse").string_value;
							if (scene.textures.find(diffuseTexName) != scene.textures.end()) {
								const tinygltf::Texture &tex = scene.textures.at(diffuseTexName);
								if (scene.images.find(tex.source) != scene.images.end()) {
									const tinygltf::Image &image = scene.images.at(tex.source);

									size_t s = image.image.size() * sizeof(TextureData);
									cudaMalloc(&dev_diffuseTex, s);
									cudaMemcpy(dev_diffuseTex, &image.image.at(0), s, cudaMemcpyHostToDevice);
									
									diffuseTexWidth = image.width;
									diffuseTexHeight = image.height;

									checkCUDAError("Set Texture Image data");
								}
							}
							/*
							if (mat.values.find("specular") != mat.values.end()) {
								std::string diffuseTexName = mat.values.at("diffuse").string_value;
								if (scene.textures.find(diffuseTexName) != scene.textures.end()) {
									const tinygltf::Texture &tex = scene.textures.at(diffuseTexName);
									if (scene.images.find(tex.source) != scene.images.end()) {
										const tinygltf::Image &image = scene.images.at(tex.source);

										size_t s = image.image.size() * sizeof(TextureData);
										cudaMalloc(&dev_diffuseTex, s);
										cudaMemcpy(dev_diffuseTex, &image.image.at(0), s, cudaMemcpyHostToDevice);

										diffuseTexWidth = image.width;
										diffuseTexHeight = image.height;

										checkCUDAError("Set Texture Image data");
									}
								}*/
						}

						// TODO: write your code for other materails
						// You may have to take a look at tinygltfloader
						// You can also use the above code loading diffuse material as a start point 
					}


					// ---------Node hierarchy transform--------
					cudaDeviceSynchronize();
					
					dim3 numBlocksNodeTransform((numVertices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
					_nodeMatrixTransform << <numBlocksNodeTransform, numThreadsPerBlock >> > (
						numVertices,
						dev_position,
						dev_normal,
						matrix,
						matrixNormal);

					checkCUDAError("Node hierarchy transformation");

					// at the end of the for loop of primitive
					// push dev pointers to map
					primitiveVector.push_back(PrimitiveDevBufPointers{
						primitive.mode,
						primitiveType,
						numPrimitives,
						numIndices,
						numVertices,

						dev_indices,
						dev_position,
						dev_normal,
						dev_texcoord0,

						dev_diffuseTex,
						diffuseTexWidth,
						diffuseTexHeight,

						dev_vertexOut	//VertexOut
					});

					totalNumPrimitives += numPrimitives;

				} // for each primitive

			} // for each mesh

		} // for each node

	}
	

	// 3. Malloc for dev_primitives
	{
		cudaMalloc(&dev_primitives, totalNumPrimitives * sizeof(Primitive));
	}
	

	// Finally, cudaFree raw dev_bufferViews
	{

		std::map<std::string, BufferByte*>::const_iterator it(bufferViewDevPointers.begin());
		std::map<std::string, BufferByte*>::const_iterator itEnd(bufferViewDevPointers.end());
			
			//bufferViewDevPointers

		for (; it != itEnd; it++) {
			cudaFree(it->second);
		}

		checkCUDAError("Free BufferView Device Mem");
	}

	// Generate Noise Texture for AO
	glm::vec2 *noise;
	noise = (glm::vec2 *)malloc(8 * 8 * sizeof(glm::vec2));
	for (int i = 0; i < 64; i++)
	{
		std::mt19937_64 rng;
		uint64_t timeSeed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
		std::seed_seq ss{ uint32_t(timeSeed & 0xffffffff), uint32_t(timeSeed >> 32) };
		rng.seed(ss);
		std::uniform_real_distribution<double> unif(0, 1);

		float x = unif(rng);
		float y = unif(rng);
		noise[i] = glm::vec2(x, y);
	}

	cudaMalloc(&dev_noise, 8 * 8 * sizeof(glm::vec2));
	cudaMemcpy(dev_noise, noise, 8 * 8 * sizeof(glm::vec2), cudaMemcpyHostToDevice);

	delete noise;


	// Init Tiles
	cudaMalloc(&dev_tile, numTiles * totalNumPrimitives * sizeof(int));

}



__global__ 
void _vertexTransformAndAssembly(
	int numVertices, 
	PrimitiveDevBufPointers primitive, 
	glm::mat4 MVP, glm::mat4 MV, glm::mat3 MV_normal, 
	int width, int height, int t) {

	// vertex id
	int vid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (vid < numVertices) {

		// TODO: Apply vertex transformation here
		// Multiply the MVP matrix for each vertex position, this will transform everything into clipping space
		// Then divide the pos by its w element to transform into NDC space
		// Finally transform x and y to viewport space
		glm::vec4 vertex_position4 = glm::vec4(primitive.dev_position[vid], 1.0f);

#if DEMO
		glm::mat4 rot(1);
		rot = glm::rotate(rot, t / 64.0f, glm::vec3(0, 1, 0));
		vertex_position4 = vertex_position4 * rot;
#endif

		glm::vec4 vertex_proj_pos = MVP * vertex_position4;
		vertex_proj_pos = vertex_proj_pos / vertex_proj_pos.w;

		float x = (1.0f - vertex_proj_pos.x) * 0.5f * width;
		float y = (1.0f - vertex_proj_pos.y) * 0.5f * height;
		float z = vertex_proj_pos.z;
		primitive.dev_verticesOut[vid].pos = glm::vec4(x, y, z, 1.0f);
		primitive.dev_verticesOut[vid].eyePos = multiplyMV(MV, vertex_position4);
		primitive.dev_verticesOut[vid].eyeNor = glm::normalize(MV_normal * primitive.dev_normal[vid]);
		primitive.dev_verticesOut[vid].texHeight = primitive.diffuseTexHeight;
		primitive.dev_verticesOut[vid].texWidth = primitive.diffuseTexWidth;
		primitive.dev_verticesOut[vid].dev_diffuseTex = primitive.dev_diffuseTex;
		if (primitive.dev_texcoord0 != NULL)
			primitive.dev_verticesOut[vid].texcoord0 = primitive.dev_texcoord0[vid];

		if(vid % 3 == 0)
			primitive.dev_verticesOut[vid].col = glm::vec3(0.8, 0.8, 0.8);
		if (vid % 3 == 1)
			primitive.dev_verticesOut[vid].col = glm::vec3(0.8, 0.8, 0.8);
		if (vid % 3 == 2)
			primitive.dev_verticesOut[vid].col = glm::vec3(0.8, 0.8, 0.8);

		// TODO: Apply vertex assembly here
		// Assemble all attribute arraies into the primitive array
		
	}
}



static int curPrimitiveBeginId = 0;

__global__ 
void _primitiveAssembly(int numIndices, int curPrimitiveBeginId, Primitive* dev_primitives, PrimitiveDevBufPointers primitive) {

	// index id
	int iid = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (iid < numIndices) {

		// TODO: uncomment the following code for a start
		// This is primitive assembly for triangles

		int pid;	// id for cur primitives vector
		if (primitive.primitiveMode == TINYGLTF_MODE_TRIANGLES) {
			pid = iid / (int)primitive.primitiveType;
			dev_primitives[pid + curPrimitiveBeginId].v[iid % (int)primitive.primitiveType]
				= primitive.dev_verticesOut[primitive.dev_indices[iid]];
			dev_primitives[pid + curPrimitiveBeginId].tex = primitive.dev_diffuseTex;
			dev_primitives[pid + curPrimitiveBeginId].size[0] = primitive.diffuseTexWidth;
			dev_primitives[pid + curPrimitiveBeginId].size[1] = primitive.diffuseTexHeight;
		}
		// TODO: other primitive types (point, line)
	}
	
}


// BackFace Culling
// https://en.wikipedia.org/wiki/Back-face_culling

__host__ __device__
bool front(glm::vec3 tri[3], int mode = 1)
{
	float z = (tri[1].x - tri[0].x) * (tri[2].y - tri[0].y) - (tri[1].y - tri[0].y) * (tri[2].x - tri[0].x);
	return mode ? z > 0.0f : z < 0.0f;
}

__host__ __device__
float sum(glm::vec3 v)
{
	return v.x + v.y + v.z;
}

__host__ __device__
glm::vec3 eval(glm::vec3 barycentric, glm::vec3 *val)
{
	return barycentric.x * val[0] + barycentric.y * val[1] + barycentric.z * val[2];
}

__host__ __device__
glm::vec2 eval(glm::vec3 barycentric, glm::vec2 *val)
{
	return barycentric.x * val[0] + barycentric.y * val[1] + barycentric.z * val[2];
}

__host__ __device__
float eval(glm::vec3 barycentric, float *val)
{
	return barycentric.x * val[0] + barycentric.y * val[1] + barycentric.z * val[2];
}

__host__ __device__
float getCorrectedZ(const glm::vec3 barycentric, const float *z) 
{
	return 1.0f / (barycentric.x / z[0] + barycentric.y / z[1] + barycentric.z / z[2]);
}
__host__ __device__
void drawWireframe(Fragment * dev_fragmentBuffer, int w, int h, glm::vec3 *tris)
{

}

// atomicMin for Float value
// method 'factomicMin' from user hyqneuron
// https://devtalk.nvidia.com/default/topic/492068/atomicmin-with-float/

__device__
float fatomicMin(float *addr, float value)
{
	float old = *addr, assumed;
	if (old <= value) return old;
	do {
		assumed = old;
		old = atomicCAS((unsigned int*)addr, __float_as_int(assumed), __float_as_int(value));
	} while (old != assumed);
	return old;
}

__global__
void _rasterization(Fragment * dev_fragmentBuffer, Primitive * dev_primitives, int nPrimatives, float * dev_depth, int width, int height)
{
	int pid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (pid < nPrimatives) {
		Primitive p = dev_primitives[pid];
		
		glm::vec3 tri[3] = {glm::vec3(p.v[0].pos), glm::vec3(p.v[1].pos), glm::vec3(p.v[2].pos)};
		glm::vec2 tex[3] = {p.v[0].texcoord0 , p.v[1].texcoord0 , p.v[2].texcoord0};
		glm::vec3 col[3] = {p.v[0].col , p.v[1].col , p.v[2].col};
		glm::vec3 eyenor[3] = { p.v[0].eyeNor, p.v[1].eyeNor, p.v[2].eyeNor };
		glm::vec3 eyepos[3] = { p.v[0].eyePos, p.v[1].eyePos, p.v[2].eyePos };
		
#if BACKFACE_CULLING
		if (!front(tri)) return;
#endif

		AABB bbox = getAABBForTriangle(tri);
		int maxx = MIN(bbox.max.x, width - 1);
		int maxy = MIN(bbox.max.y, height - 1);
		int minx = MAX(bbox.min.x, 0);
		int miny = MAX(bbox.min.y, 0);
		
	#if !SHADING
		for(int x = minx; x <= maxx; x++)
		for(int y = miny; y <= maxy; y++)
		{
			glm::vec2 xy = glm::vec2(x, y);
			glm::vec3 barycentric = calculateBarycentricCoordinate(tri, xy);
			if (isBarycentricCoordInBounds(barycentric)) {
				float depth = getZAtCoordinate(barycentric, tri) * (float)INT_MIN;
				int index = x + y * width;
				fatomicMin(&dev_depth[index], depth);
				if (depth == dev_depth[index]) {
					dev_fragmentBuffer[index].z = depth;
					dev_fragmentBuffer[index].color = eval(barycentric, col);
					dev_fragmentBuffer[index].eyeNor = glm::normalize(eval(barycentric, eyenor));
					dev_fragmentBuffer[index].eyePos = eval(barycentric, eyepos);
					if (dev_primitives[pid].tex != NULL) {
						dev_fragmentBuffer[index].dev_diffuseTex = dev_primitives[pid].tex;
						dev_fragmentBuffer[index].size[0] = dev_primitives[pid].size[0];
						dev_fragmentBuffer[index].size[1] = dev_primitives[pid].size[1];
					
						// Perspective Correted Texture Coord
						float c[3] = {  eyepos[0].z,  eyepos[1].z, eyepos[2].z };
						glm::vec2 ttex[3] = { tex[0] / eyepos[0].z, tex[1] / eyepos[1].z, tex[2] / eyepos[2].z };
						float cz = getCorrectedZ(barycentric, c);
						dev_fragmentBuffer[index].texcoord0 = cz * eval(barycentric, ttex);

					}
				}

			}
		}

	#else
		for (int x = minx; x <= maxx; x++)
			for (int y = miny; y <= maxy; y++)
			{
				glm::vec2 xy = glm::vec2(x, y);
				glm::vec3 barycentric = calculateBarycentricCoordinate(tri, xy);
				#if SHADING == 1
					bool mode = isBarycentricCoordOnBounds(barycentric);
				#endif
				#if SHADING == 2
					bool mode = isBarycentricCoordOnCorner(barycentric);
				#endif
				if (mode) {
					float depth = getZAtCoordinate(barycentric, tri) * (float)INT_MIN;
					int index = x + y * width;
					fatomicMin(&dev_depth[index], depth);
					if (depth == dev_depth[index]) {
						dev_fragmentBuffer[index].z = depth;
						#if SHADING == 1
							dev_fragmentBuffer[index].color = glm::vec3(0.1, 1, 0.1);
						#endif
						#if SHADING == 2
							dev_fragmentBuffer[index].color = glm::vec3(1, 0.1, 0.1);
						#endif
						dev_fragmentBuffer[index].eyeNor = glm::normalize(eval(barycentric, eyenor));
						dev_fragmentBuffer[index].eyePos = eval(barycentric, eyepos);
						dev_fragmentBuffer[index].dev_diffuseTex = NULL;
					}

				}
			}
	#endif
	}
}

__host__ __device__
void tileBound(AABB bbox, int *minmax, glm::vec3 *tri, glm::vec2 tilewh, int tilesize)
{
	int w = tilewh.x;
	int h = tilewh.y;

	minmax[0] = MIN(floorf(bbox.max.x + 0.5f) / tilesize, w - 1);
	minmax[1] = MIN(floorf(bbox.max.y + 0.5f) / tilesize, h - 1);
	minmax[2] = MAX((floorf(bbox.min.x + 0.5f) / tilesize) - 0, 0);
	minmax[3] = MAX((floorf(bbox.min.y + 0.5f) / tilesize) - 0, 0);
}


// tilesize: 
__global__
void _decompose(Primitive * dev_primitives, int * count, int nPrimatives, int *tile, glm::vec2 tileWidthHeight, int tilesize)
{
	int pid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (pid < nPrimatives) {
		Primitive p = dev_primitives[pid];
		glm::vec3 tri[3] = { glm::vec3(p.v[0].pos), glm::vec3(p.v[1].pos), glm::vec3(p.v[2].pos) };

#if BACKFACE_CULLING
		if (!front(tri)) return;
#endif

		int bound[4];
		AABB bbox = getAABBForTriangle(tri);
		tileBound(bbox, bound, tri, tileWidthHeight, tilesize);
		for (int i = bound[2]; i <= bound[0]; i++)
		for (int j = bound[3]; j <= bound[1]; j++)
		{
			int w = tileWidthHeight.x;
			int h = tileWidthHeight.y;
			int off = atomicAdd(&count[i + j * w], 1);
			tile[(i + j * w) * nPrimatives + off] = pid;
		}
	}
}


__global__
void _rasterizationWithTiles(Fragment * dev_fragmentBuffer, Primitive * dev_primitives, int nPrimatives, float * dev_depth, 
							  int width, int height, int *count, int *tile, glm::vec2 tileWidthHeight, int tilesize)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	if (x >= 0 && y >= 0 && x < width && y < height)
	{
		int idx = blockIdx.x + blockIdx.y * tileWidthHeight.x; // tile index
		int index = x + y * width; // fragment index

		for (int i = 0; i < count[idx]; i++) {
			int pid = tile[idx * nPrimatives + i];
			//printf("pid: %i\n", count[idx]);
			Primitive p = dev_primitives[pid];
			glm::vec3 tri[3] = { glm::vec3(p.v[0].pos), glm::vec3(p.v[1].pos), glm::vec3(p.v[2].pos) };
			glm::vec2 tex[3] = { p.v[0].texcoord0 , p.v[1].texcoord0 , p.v[2].texcoord0 };
			glm::vec3 col[3] = { p.v[0].col , p.v[1].col , p.v[2].col };
			glm::vec3 eyenor[3] = { p.v[0].eyeNor, p.v[1].eyeNor, p.v[2].eyeNor };
			glm::vec3 eyepos[3] = { p.v[0].eyePos, p.v[1].eyePos, p.v[2].eyePos };

			glm::vec2 xy = glm::vec2(x, y);
			glm::vec3 barycentric = calculateBarycentricCoordinate(tri, xy);
			if (isBarycentricCoordInBounds(barycentric)) {
				float depth = getZAtCoordinate(barycentric, tri) * (float)INT_MIN;
				
				fatomicMin(&dev_depth[index], depth);
				if (depth == dev_depth[index]) {
					dev_fragmentBuffer[index].z = depth;
					dev_fragmentBuffer[index].color = eval(barycentric, col);
					dev_fragmentBuffer[index].eyeNor = glm::normalize(eval(barycentric, eyenor));
					dev_fragmentBuffer[index].eyePos = eval(barycentric, eyepos);
					if (dev_primitives[pid].tex != NULL) {
						dev_fragmentBuffer[index].dev_diffuseTex = dev_primitives[pid].tex;
						dev_fragmentBuffer[index].size[0] = dev_primitives[pid].size[0];
						dev_fragmentBuffer[index].size[1] = dev_primitives[pid].size[1];

						// Perspective Correted Texture Coord
						float c[3] = { eyepos[0].z,  eyepos[1].z, eyepos[2].z };
						glm::vec2 ttex[3] = { tex[0] / eyepos[0].z, tex[1] / eyepos[1].z, tex[2] / eyepos[2].z };
						float cz = getCorrectedZ(barycentric, c);
						dev_fragmentBuffer[index].texcoord0 = cz * eval(barycentric, ttex);

					}
				}

			}
		}
	}

}


// Blur based on ...
// https://www.slideshare.net/DarshanParsana/gaussian-image-blurring-in-cuda-c
__global__
void Blur(int width, int height, glm::vec3 *dev_framebuffer, float a = 1.0f, bool th = false) {
	const int R = 2;
	const int bw = 32;
	const int bh = 32;

	int x = blockIdx.x * (bw - 2 * R) + threadIdx.x;
	int y = blockIdx.y * (bh - 2 * R) + threadIdx.y;

	x = clamp(x, 0, width - 1);
	y = clamp(y, 0, height - 1);

	int idx = threadIdx.x + threadIdx.y * blockDim.x;
	int tidx = x + y * width;

	__shared__ glm::vec3 sm[bw * bh];
	sm[idx] = dev_framebuffer[tidx];
	__syncthreads();

		if (threadIdx.x >= R && threadIdx.x < (bw - R) && threadIdx.y >= R && threadIdx.y < (bh - R)) {
			glm::vec3 sum(0);
			for (int dy = -R; dy <= R; dy++)
				for (int dx = -R; dx <= R; dx++)
				{
					glm::vec3 i = sm[idx + (dy * blockDim.y) + dx];
					sum += float(mx[dy][dx]) * i;
				}
			if(th) dev_framebuffer[tidx] = (a * ((sm[idx].r + sm[idx].g + sm[idx].b) / 3.0f) * sum / 128.0f) + (1 - a) * dev_framebuffer[tidx];
			else dev_framebuffer[tidx] = a * sum / 128.0f;
		}
}

__global__
void copyToChannel(int width, int height, glm::vec3 *frame, float * arr, int channel)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= 0 && y >= 0 && x < width && y < height)
	{
		if (channel == 0) arr[x + y * width] = frame[x + y * width].r;
		if (channel == 1) arr[x + y * width] = frame[x + y * width].g;
		if (channel == 2) arr[x + y * width] = frame[x + y * width].b;
		//printf("r: %f\n", frame[x + y * width].r);
	}
}

/**
 * Perform rasterization.
 */
void rasterize(uchar4 *pbo, const glm::mat4 & MVP, const glm::mat4 & MV, const glm::mat3 MV_normal, float time) {
    int sideLength2d = 16;
    dim3 blockSize2d(sideLength2d, sideLength2d);
    dim3 blockCount2d((width - 1) / blockSize2d.x + 1,
		(height - 1) / blockSize2d.y + 1);

	_time = time;

	// Execute your rasterization pipeline here
	// (See README for rasterization pipeline outline.)

	// Vertex Process & primitive assembly
	{
		curPrimitiveBeginId = 0;
		dim3 numThreadsPerBlock(128);

		auto it = mesh2PrimitivesMap.begin();
		auto itEnd = mesh2PrimitivesMap.end();

		for (; it != itEnd; ++it) {
			auto p = (it->second).begin();	// each primitive
			auto pEnd = (it->second).end();
			for (; p != pEnd; ++p) {
				dim3 numBlocksForVertices((p->numVertices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
				dim3 numBlocksForIndices((p->numIndices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);

				_vertexTransformAndAssembly << < numBlocksForVertices, numThreadsPerBlock >> >(p->numVertices, *p, MVP, MV, MV_normal, width, height, _time);
				checkCUDAError("Vertex Processing");
				cudaDeviceSynchronize();
				_primitiveAssembly << < numBlocksForIndices, numThreadsPerBlock >> >
					(p->numIndices, 
					curPrimitiveBeginId, 
					dev_primitives, 
					*p);
				checkCUDAError("Primitive Assembly");

				curPrimitiveBeginId += p->numPrimitives;
			}
		}

		checkCUDAError("Vertex Processing and Primitive Assembly");
	}
	
	cudaMemset(dev_fragmentBuffer, 0, width * height * sizeof(Fragment));
	initDepth << <blockCount2d, blockSize2d >> >(width, height, dev_depth);


#if !TILE_BASED_RENDER
	
	const int THREADS = 128;
	dim3 threadsPerBlock(THREADS);
	dim3 blocksPerGrid((totalNumPrimitives + THREADS - 1) / THREADS);
	cudaFuncSetCacheConfig(_rasterization, cudaFuncCachePreferL1);
	_rasterization << <blocksPerGrid, threadsPerBlock >> > (dev_fragmentBuffer, dev_primitives, totalNumPrimitives, dev_depth, width, height);
	checkCUDAError("rasteration");
	
#else
	cudaMemset(dev_count, 0, numTiles * sizeof(int));
	dim3 tsize(128);
	dim3 bsize((totalNumPrimitives + 128 - 1) / 128);
	_decompose <<<bsize, tsize >>>(dev_primitives, dev_count, totalNumPrimitives, dev_tile, glm::vec2(numTilesW, numTilesH), tilesize);
	checkCUDAError("Decompose");

	bsize = dim3(numTilesW, numTilesH, 1);
	tsize = dim3(tilesize, tilesize, 1);
	cudaFuncSetCacheConfig(_rasterizationWithTiles, cudaFuncCachePreferL1);
	_rasterizationWithTiles << <bsize, tsize >> >(dev_fragmentBuffer, dev_primitives, totalNumPrimitives, dev_depth, width, height, dev_count, dev_tile, glm::vec2(numTilesW, numTilesH), tilesize);
	checkCUDAError("Rasterization With Tiles");
#endif

    // Copy depthbuffer colors into framebuffer
	int shareMemSize = sideLength2d * sideLength2d * sizeof(glm::vec3);
	cudaFuncSetCacheConfig(render, cudaFuncCachePreferL1);
	render <<<blockCount2d, blockSize2d, shareMemSize >>>(width, height, dev_fragmentBuffer, dev_framebuffer, dev_noise);
	checkCUDAError("fragment shader");
	
#if BLOOM

	const int  t = 32;
	const int _t = 32;
	dim3 _tsize(_t, _t, 1);
	dim3 _bsize(width / t, height / t, 1);
	Blur <<<_bsize, _tsize >>> (width, height, dev_framebuffer, 0.5f, true);
	Blur << <_bsize, _tsize >> > (width, height, dev_framebuffer, 0.75f, true);
	Blur <<<_bsize, _tsize >>> (width, height, dev_framebuffer, 1.25f);
	//Blur << <_bsize, _tsize >> > (width, height, dev_framebuffer, 1.25f);

#endif
	
	
    // Copy framebuffer into OpenGL buffer for OpenGL previewing
    sendImageToPBO<<<blockCount2d, blockSize2d>>>(pbo, ow, oh, dev_framebuffer);
    checkCUDAError("copy render result to pbo");
}

/**
 * Called once at the end of the program to free CUDA memory.
 */
void rasterizeFree() {

    // deconstruct primitives attribute/indices device buffer

	auto it(mesh2PrimitivesMap.begin());
	auto itEnd(mesh2PrimitivesMap.end());
	for (; it != itEnd; ++it) {
		for (auto p = it->second.begin(); p != it->second.end(); ++p) {
			cudaFree(p->dev_indices);
			cudaFree(p->dev_position);
			cudaFree(p->dev_normal);
			cudaFree(p->dev_texcoord0);
			cudaFree(p->dev_diffuseTex);

			cudaFree(p->dev_verticesOut);

			
			//TODO: release other attributes and materials
		}
	}

	////////////

    cudaFree(dev_primitives);
    dev_primitives = NULL;

	cudaFree(dev_fragmentBuffer);
	dev_fragmentBuffer = NULL;

    cudaFree(dev_framebuffer);
    dev_framebuffer = NULL;

	cudaFree(dev_depth);
	dev_depth = NULL;

    checkCUDAError("rasterize Free");
}
