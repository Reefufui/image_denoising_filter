#include <vulkan/vulkan.h>

#include <vector>
#include <cstring>
#include <cassert>
#include <stdexcept>
#include <cmath>

#include "Bitmap.h" // Save bmp file

#include <iostream>

// const int WIDTH          = 3200;
// const int HEIGHT         = 2400;
// const int WORKGROUP_SIZE = 16;

const int WIDTH          = 4096;
const int HEIGHT         = 4096;
const int WORKGROUP_SIZE = 16;


#ifdef NDEBUG
constexpr bool enableValidationLayers = false;
#else
constexpr bool enableValidationLayers = true;
#endif

#include "vk_utils.h"

struct CustomVulkanTexture
{
    CustomVulkanTexture() : imagesMemoryGPU(0), imageGPU(0), imageSampler(0), imageView(0) {}

    void Release(VkDevice a_device)
    {
        vkFreeMemory      (a_device, imagesMemoryGPU, NULL);
        vkDestroyImage    (a_device, imageGPU,        NULL);
        vkDestroyImageView(a_device, imageView,       NULL);
        vkDestroySampler  (a_device, imageSampler,    NULL);
    }

    VkDeviceMemory imagesMemoryGPU; 
    // TODO: this is bad design in fact, you should store memory somewhere else and/or use memory pools;
    VkImage        imageGPU;
    VkSampler      imageSampler;
    VkImageView    imageView;

    static CustomVulkanTexture Create2DTextureRGBA256(VkDevice a_device, VkPhysicalDevice a_physDevice, int w, int h);

    static void CreateTexture(VkDevice a_device, VkPhysicalDevice a_physDevice, const int a_width, const int a_height,
            VkImage *a_images, VkDeviceMemory *a_pImagesMemory)
    {
        VkImageCreateInfo imgCreateInfo = {};
        imgCreateInfo.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imgCreateInfo.pNext         = nullptr;
        imgCreateInfo.flags         = 0; // not sure about this ...
        imgCreateInfo.imageType     = VK_IMAGE_TYPE_2D;
        imgCreateInfo.format        = VK_FORMAT_R8G8B8A8_UNORM;
        imgCreateInfo.extent        = VkExtent3D{uint32_t(a_width), uint32_t(a_height), 1};
        imgCreateInfo.mipLevels     = 1;
        imgCreateInfo.samples       = VK_SAMPLE_COUNT_1_BIT;
        imgCreateInfo.tiling        = VK_IMAGE_TILING_OPTIMAL;
        imgCreateInfo.usage         = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        imgCreateInfo.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
        imgCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        imgCreateInfo.arrayLayers   = 1;
        VK_CHECK_RESULT(vkCreateImage(a_device, &imgCreateInfo, nullptr, a_images + 0));

        VkMemoryRequirements memoryRequirements;
        vkGetImageMemoryRequirements(a_device, a_images[0], &memoryRequirements);
        VkMemoryAllocateInfo allocateInfo = {};
        allocateInfo.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocateInfo.allocationSize  = memoryRequirements.size;
        allocateInfo.memoryTypeIndex = vk_utils::FindMemoryType(memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, a_physDevice);
        VK_CHECK_RESULT(vkAllocateMemory(a_device, &allocateInfo, NULL, a_pImagesMemory));
        VK_CHECK_RESULT(vkBindImageMemory(a_device, a_images[0], (*a_pImagesMemory), 0));
    }

    void CreateOther(VkDevice a_device)
    {
        VkSamplerCreateInfo samplerInfo = {};
        {
            samplerInfo.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
            samplerInfo.pNext        = nullptr;
            samplerInfo.flags        = 0;
            samplerInfo.magFilter    = VK_FILTER_LINEAR;
            samplerInfo.minFilter    = VK_FILTER_LINEAR;
            samplerInfo.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
            samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
            samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
            samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
            samplerInfo.mipLodBias   = 0.0f;
            samplerInfo.compareOp    = VK_COMPARE_OP_NEVER;
            samplerInfo.minLod           = 0;
            samplerInfo.maxLod           = 0;
            samplerInfo.maxAnisotropy    = 1.0;
            samplerInfo.anisotropyEnable = VK_FALSE;
            samplerInfo.borderColor      = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
            samplerInfo.unnormalizedCoordinates = VK_FALSE;
        }
        VK_CHECK_RESULT(vkCreateSampler(a_device, &samplerInfo, nullptr, &this->imageSampler));

        VkImageViewCreateInfo imageViewInfo = {};
        {
            imageViewInfo.sType      = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            imageViewInfo.flags      = 0;
            imageViewInfo.viewType   = VK_IMAGE_VIEW_TYPE_2D;
            imageViewInfo.format     = VK_FORMAT_R8G8B8A8_UNORM;
            imageViewInfo.components = { VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G, VK_COMPONENT_SWIZZLE_B, VK_COMPONENT_SWIZZLE_A };
            imageViewInfo.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
            imageViewInfo.subresourceRange.baseMipLevel   = 0;
            imageViewInfo.subresourceRange.baseArrayLayer = 0;
            imageViewInfo.subresourceRange.layerCount     = 1;
            imageViewInfo.subresourceRange.levelCount     = 1;
            imageViewInfo.image = this->imageGPU;
        }
        VK_CHECK_RESULT(vkCreateImageView(a_device, &imageViewInfo, nullptr, &this->imageView));
    }


};

CustomVulkanTexture CustomVulkanTexture::Create2DTextureRGBA256(VkDevice a_device, VkPhysicalDevice a_physDevice, int w, int h)
{
    CustomVulkanTexture res{};

    res.CreateTexture(a_device, a_physDevice, w, h, &res.imageGPU, &res.imagesMemoryGPU);
    // (w,h) ==> (imageGPU, imagesMemoryGPU); for R8G8B8A8_UNORM format

    res.CreateOther(a_device);

    return res;
}

class ComputeApplication
{
    private:

        struct Pixel {
            float r, g, b, a;
        };

        VkInstance                m_instance{};
        VkDebugReportCallbackEXT  m_debugReportCallback{};
        VkPhysicalDevice          m_physicalDevice{};
        VkDevice                  m_device{};
        VkPipeline                m_pipeline{};
        VkPipelineLayout          m_pipelineLayout{};
        VkShaderModule            m_computeShaderModule{};
        VkCommandPool             m_commandPool{};
        VkCommandBuffer           m_commandBuffer{};
        VkDescriptorPool          m_descriptorPool{};
        VkDescriptorSet           m_descriptorSet{};
        VkDescriptorSetLayout     m_descriptorSetLayout{};
        VkBuffer                  m_bufferGPU{}, m_bufferStaging{}, m_bufferDynamic{};
        VkDeviceMemory            m_bufferMemoryGPU{}, m_bufferMemoryStaging{}, m_bufferMemoryDynamic{};
        CustomVulkanTexture       m_img{};
        std::vector<const char *> m_enabledLayers{};
        VkQueue                   m_queue{};

    public:

        ComputeApplication() : m_bufferDynamic(NULL), m_bufferMemoryDynamic(NULL) { }

        static void GetImageFromGPU(VkDevice a_device, VkDeviceMemory a_stagingMem, int a_w, int a_h, uint32_t *a_imageData)
        {
            void *mappedMemory = nullptr;
            vkMapMemory(a_device, a_stagingMem, 0, a_w * a_h * sizeof(float) * 4, 0, &mappedMemory);
            Pixel* pmappedMemory = (Pixel *)mappedMemory;

            for (int i = 0; i < a_w * a_h; ++i)
            {
                const uint32_t r = ((uint32_t) (255.0f * (pmappedMemory[i].r)));
                const uint32_t g = ((uint32_t) (255.0f * (pmappedMemory[i].g)));
                const uint32_t b = ((uint32_t) (255.0f * (pmappedMemory[i].b)));
                a_imageData[i] = (r << 0) | (g << 8) | (b << 16);
            }
            
            vkUnmapMemory(a_device, a_stagingMem);
        }

        static void PutImageToGPU(VkDevice a_device, VkDeviceMemory a_dynamicMem, int a_w, int a_h, const uint32_t *a_imageData)
        {
            void *mappedMemory = nullptr;
            vkMapMemory(a_device, a_dynamicMem, 0, a_w * a_h * sizeof(float) * 4, 0, &mappedMemory);
            float* pmappedMemory = (float*)mappedMemory;
            for (int i = 0; i < (a_w * a_h); i ++)
            {
                const uint32_t b = (a_imageData[i] & 0x00FF0000) >> 16;
                const uint32_t g = (a_imageData[i] & 0x0000FF00) >> 8;
                const uint32_t r = (a_imageData[i] & 0x000000FF);

                pmappedMemory[i*4+0] = float(r)*(1.0f/255.0f);
                pmappedMemory[i*4+1] = float(g)*(1.0f/255.0f);
                pmappedMemory[i*4+2] = float(b)*(1.0f/255.0f);
                pmappedMemory[i*4+3] = 0.0f;
            }

            vkUnmapMemory(a_device, a_dynamicMem);
        }


        static VKAPI_ATTR VkBool32 VKAPI_CALL debugReportCallbackFn(
                VkDebugReportFlagsEXT                       flags,
                VkDebugReportObjectTypeEXT                  objectType,
                uint64_t                                    object,
                size_t                                      location,
                int32_t                                     messageCode,
                const char*                                 pLayerPrefix,
                const char*                                 pMessage,
                void*                                       pUserData)
        {
            printf("Debug Report: %s: %s\n", pLayerPrefix, pMessage);
            return VK_FALSE;
        }


        // CPU
        static void CreateStagingBuffer(VkDevice a_device, VkPhysicalDevice a_physDevice, const size_t a_bufferSize,
                VkBuffer *a_pBuffer, VkDeviceMemory *a_pBufferMemory)
        {
            VkBufferCreateInfo bufferCreateInfo{};
            bufferCreateInfo.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            bufferCreateInfo.size        = a_bufferSize;
            bufferCreateInfo.usage       = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            //  #NOTE bufferStaging is used as a storage bufferStaging and we can _copy_to_ it

            VK_CHECK_RESULT(vkCreateBuffer(a_device, &bufferCreateInfo, NULL, a_pBuffer));

            VkMemoryRequirements memoryRequirements{};
            vkGetBufferMemoryRequirements(a_device, (*a_pBuffer), &memoryRequirements);

            VkMemoryAllocateInfo allocateInfo{};
            allocateInfo.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            allocateInfo.allocationSize  = memoryRequirements.size;
            allocateInfo.memoryTypeIndex = vk_utils::FindMemoryType(
                    memoryRequirements.memoryTypeBits,
                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                    a_physDevice);

            VK_CHECK_RESULT(vkAllocateMemory(a_device, &allocateInfo, NULL, a_pBufferMemory));

            VK_CHECK_RESULT(vkBindBufferMemory(a_device, (*a_pBuffer), (*a_pBufferMemory), 0));
        }


        //GPU
        static void CreateDynamicBuffer(VkDevice a_device, VkPhysicalDevice a_physDevice, const size_t a_bufferSize,
                VkBuffer *a_pBuffer, VkDeviceMemory *a_pBufferMemory)
        {
            VkBufferCreateInfo bufferCreateInfo{};
            bufferCreateInfo.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            bufferCreateInfo.size        = a_bufferSize;
            bufferCreateInfo.usage       = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
            bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

            VK_CHECK_RESULT(vkCreateBuffer(a_device, &bufferCreateInfo, NULL, a_pBuffer));

            VkMemoryRequirements memoryRequirements;
            vkGetBufferMemoryRequirements(a_device, (*a_pBuffer), &memoryRequirements);

            VkMemoryAllocateInfo allocateInfo{};
            allocateInfo.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            allocateInfo.allocationSize  = memoryRequirements.size;
            allocateInfo.memoryTypeIndex = vk_utils::FindMemoryType(
                    memoryRequirements.memoryTypeBits,
                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                    a_physDevice);

            VK_CHECK_RESULT(vkAllocateMemory(a_device, &allocateInfo, NULL, a_pBufferMemory));

            VK_CHECK_RESULT(vkBindBufferMemory(a_device, (*a_pBuffer), (*a_pBufferMemory), 0));
        }


        static void CreateWriteOnlyBuffer(VkDevice a_device, VkPhysicalDevice a_physDevice, const size_t a_bufferSize,
                VkBuffer *a_pBuffer, VkDeviceMemory *a_pBufferMemory)
        {
            VkBufferCreateInfo bufferCreateInfo{};
            bufferCreateInfo.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            bufferCreateInfo.size        = a_bufferSize;
            bufferCreateInfo.usage       = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
            bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

            VK_CHECK_RESULT(vkCreateBuffer(a_device, &bufferCreateInfo, NULL, a_pBuffer));

            VkMemoryRequirements memoryRequirements;
            vkGetBufferMemoryRequirements(a_device, (*a_pBuffer), &memoryRequirements);


            VkMemoryAllocateInfo allocateInfo = {};
            allocateInfo.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            allocateInfo.allocationSize  = memoryRequirements.size;
            allocateInfo.memoryTypeIndex = vk_utils::FindMemoryType(
                    memoryRequirements.memoryTypeBits,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                    a_physDevice);

            VK_CHECK_RESULT(vkAllocateMemory(a_device, &allocateInfo, NULL, a_pBufferMemory));

            VK_CHECK_RESULT(vkBindBufferMemory(a_device, (*a_pBuffer), (*a_pBufferMemory), 0));
        }

        static void CreateDescriptorSetLayout(VkDevice a_device, VkDescriptorSetLayout *a_pDSLayout)
        {
            VkDescriptorSetLayoutBinding descriptorSetLayoutBinding[2];
            descriptorSetLayoutBinding[0].binding            = 0;
            descriptorSetLayoutBinding[0].descriptorType     = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptorSetLayoutBinding[0].descriptorCount    = 1;
            descriptorSetLayoutBinding[0].stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT;
            descriptorSetLayoutBinding[0].pImmutableSamplers = nullptr;

            descriptorSetLayoutBinding[1].binding            = 1;
            descriptorSetLayoutBinding[1].descriptorType     = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorSetLayoutBinding[1].descriptorCount    = 1;
            descriptorSetLayoutBinding[1].stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT;
            descriptorSetLayoutBinding[1].pImmutableSamplers = nullptr;

            VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo{};
            descriptorSetLayoutCreateInfo.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            descriptorSetLayoutCreateInfo.bindingCount = 2;
            descriptorSetLayoutCreateInfo.pBindings    = descriptorSetLayoutBinding;

            VK_CHECK_RESULT(vkCreateDescriptorSetLayout(a_device, &descriptorSetLayoutCreateInfo, NULL, a_pDSLayout));
        }

        void CreateDescriptorSetForOurBufferAndTexture(VkDevice a_device, VkBuffer a_buffer, size_t a_bufferSize, const VkDescriptorSetLayout *a_pDSLayout, VkImage a_image,
                VkDescriptorPool *a_pDSPool, VkDescriptorSet *a_pDS, VkSampler *a_samplers, VkImageView *a_views)
        {
            VkDescriptorPoolSize descriptorPoolSize[2];
            descriptorPoolSize[0].type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptorPoolSize[0].descriptorCount = 1;
            descriptorPoolSize[1].type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorPoolSize[1].descriptorCount = 1;


            VkDescriptorPoolCreateInfo descriptorPoolCreateInfo{};
            descriptorPoolCreateInfo.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            descriptorPoolCreateInfo.maxSets       = 1;
            descriptorPoolCreateInfo.poolSizeCount = 2;
            descriptorPoolCreateInfo.pPoolSizes    = descriptorPoolSize;

            VK_CHECK_RESULT(vkCreateDescriptorPool(a_device, &descriptorPoolCreateInfo, NULL, a_pDSPool));

            VkDescriptorSetAllocateInfo descriptorSetAllocateInfo{};
            descriptorSetAllocateInfo.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            descriptorSetAllocateInfo.descriptorPool     = (*a_pDSPool);
            descriptorSetAllocateInfo.descriptorSetCount = 1;
            descriptorSetAllocateInfo.pSetLayouts        = a_pDSLayout;

            VK_CHECK_RESULT(vkAllocateDescriptorSets(a_device, &descriptorSetAllocateInfo, a_pDS));

            VkDescriptorBufferInfo descriptorBufferInfo{};
            descriptorBufferInfo.buffer = a_buffer;
            descriptorBufferInfo.offset = 0;
            descriptorBufferInfo.range  = a_bufferSize;

            VkWriteDescriptorSet writeDescriptorSet{};
            writeDescriptorSet.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writeDescriptorSet.dstSet          = *(a_pDS+0);
            writeDescriptorSet.dstBinding      = 0;
            writeDescriptorSet.descriptorCount = 1;
            writeDescriptorSet.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writeDescriptorSet.pBufferInfo     = &descriptorBufferInfo;

            vkUpdateDescriptorSets(a_device, 1, &writeDescriptorSet, 0, NULL);

            VkDescriptorImageInfo descriptorImageInfo{};
            descriptorImageInfo.sampler     = m_img.imageSampler;
            descriptorImageInfo.imageView   = m_img.imageView;
            descriptorImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            VkWriteDescriptorSet writeDescriptorSet2{};
            writeDescriptorSet2.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writeDescriptorSet2.dstSet          = *(a_pDS+0);
            writeDescriptorSet2.dstBinding      = 1;
            writeDescriptorSet2.descriptorCount = 1;
            writeDescriptorSet2.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writeDescriptorSet2.pImageInfo      = &descriptorImageInfo;

            vkUpdateDescriptorSets(a_device, 1, &writeDescriptorSet2, 0, NULL);
        }

        static void CreateComputePipeline(VkDevice a_device, const VkDescriptorSetLayout &a_dsLayout,
                VkShaderModule *a_pShaderModule, VkPipeline *a_pPipeline, VkPipelineLayout *a_pPipelineLayout)
        {
            std::vector<uint32_t> code = vk_utils::ReadFile("shaders/comp.spv");
            VkShaderModuleCreateInfo createInfo{};
            createInfo.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            createInfo.pCode    = code.data();
            createInfo.codeSize = code.size()*sizeof(uint32_t);

            VK_CHECK_RESULT(vkCreateShaderModule(a_device, &createInfo, NULL, a_pShaderModule));

            VkPipelineShaderStageCreateInfo shaderStageCreateInfo = {};
            shaderStageCreateInfo.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            shaderStageCreateInfo.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
            shaderStageCreateInfo.module = (*a_pShaderModule);
            shaderStageCreateInfo.pName  = "main";

            // Allow pass (w,h) inside shader directly from command buffer
            
            VkPushConstantRange pcRange{}; // #NOTE: we updated this to pass W/H inside shader
            pcRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            pcRange.offset     = 0;
            pcRange.size       = 2 * sizeof(int); // #NOTE: (w,h); please add more memory if you need more parameters!

            VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {};
            pipelineLayoutCreateInfo.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            pipelineLayoutCreateInfo.setLayoutCount         = 1;
            pipelineLayoutCreateInfo.pSetLayouts            = &a_dsLayout;
            pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
            pipelineLayoutCreateInfo.pPushConstantRanges    = &pcRange;
            VK_CHECK_RESULT(vkCreatePipelineLayout(a_device, &pipelineLayoutCreateInfo, NULL, a_pPipelineLayout));

            VkComputePipelineCreateInfo pipelineCreateInfo = {};
            pipelineCreateInfo.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
            pipelineCreateInfo.stage  = shaderStageCreateInfo;
            pipelineCreateInfo.layout = (*a_pPipelineLayout);

            VK_CHECK_RESULT(vkCreateComputePipelines(a_device, VK_NULL_HANDLE, 1, &pipelineCreateInfo, NULL, a_pPipeline));
        }

        static void CreateCommandBuffer(VkDevice a_device, uint32_t a_queueFamilyIndex, VkPipeline a_pipeline, VkPipelineLayout a_layout,
                VkCommandPool *a_pool, VkCommandBuffer *a_pCmdBuff)
        {
            VkCommandPoolCreateInfo commandPoolCreateInfo{};
            commandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
            commandPoolCreateInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

            commandPoolCreateInfo.queueFamilyIndex = a_queueFamilyIndex;
            VK_CHECK_RESULT(vkCreateCommandPool(a_device, &commandPoolCreateInfo, NULL, a_pool));

            VkCommandBufferAllocateInfo commandBufferAllocateInfo{};
            commandBufferAllocateInfo.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            commandBufferAllocateInfo.commandPool        = (*a_pool);
            commandBufferAllocateInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            commandBufferAllocateInfo.commandBufferCount = 1;
            VK_CHECK_RESULT(vkAllocateCommandBuffers(a_device, &commandBufferAllocateInfo, a_pCmdBuff));
        }

        static VkImageMemoryBarrier imBarTransfer(VkImage a_image, const VkImageSubresourceRange& a_range, VkImageLayout before, VkImageLayout after)
        {
            VkImageMemoryBarrier moveToGeneralBar{};
            moveToGeneralBar.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            moveToGeneralBar.pNext               = nullptr;
            moveToGeneralBar.srcAccessMask       = 0;
            moveToGeneralBar.dstAccessMask       = VK_PIPELINE_STAGE_TRANSFER_BIT;
            moveToGeneralBar.oldLayout           = before;
            moveToGeneralBar.newLayout           = after;
            moveToGeneralBar.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            moveToGeneralBar.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            moveToGeneralBar.image               = a_image;
            moveToGeneralBar.subresourceRange    = a_range;
            return moveToGeneralBar;
        }

        static VkImageSubresourceRange WholeImageRange()
        {
            VkImageSubresourceRange rangeWholeImage{};
            rangeWholeImage.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
            rangeWholeImage.baseMipLevel   = 0;
            rangeWholeImage.levelCount     = 1;
            rangeWholeImage.baseArrayLayer = 0;
            rangeWholeImage.layerCount     = 1;
            return rangeWholeImage;
        }

        static void RecordCommandsOfExecuteAndTransfer(VkCommandBuffer a_cmdBuff, VkPipeline a_pipeline,VkPipelineLayout a_layout, const VkDescriptorSet &a_ds, VkImage a_image,
                size_t a_bufferSize, VkBuffer a_bufferGPU, VkBuffer a_bufferStaging)
        {
            VkCommandBufferBeginInfo beginInfo{};
            beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            VK_CHECK_RESULT(vkBeginCommandBuffer(a_cmdBuff, &beginInfo));

            vkCmdFillBuffer(a_cmdBuff, a_bufferStaging, 0, a_bufferSize, 0);
            // NOTE: clear this buffer just for an example and test cases. if we comment 'vkCmdCopyBuffer', we'll get black image

            vkCmdBindPipeline      (a_cmdBuff, VK_PIPELINE_BIND_POINT_COMPUTE, a_pipeline);
            vkCmdBindDescriptorSets(a_cmdBuff, VK_PIPELINE_BIND_POINT_COMPUTE, a_layout, 0, 1, &a_ds, 0, NULL);

            int wh[2]{ WIDTH, HEIGHT };
            vkCmdPushConstants(a_cmdBuff, a_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(int) * 2, wh);

            vkCmdDispatch(a_cmdBuff, (uint32_t)ceil(WIDTH / float(WORKGROUP_SIZE)), (uint32_t)ceil(HEIGHT / float(WORKGROUP_SIZE)), 1);

            VkBufferMemoryBarrier bufBarr = {};
            bufBarr.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            bufBarr.pNext = nullptr;
            bufBarr.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            bufBarr.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            bufBarr.size                = VK_WHOLE_SIZE;
            bufBarr.offset              = 0;
            bufBarr.buffer              = a_bufferGPU;
            bufBarr.srcAccessMask       = VK_ACCESS_SHADER_WRITE_BIT;
            bufBarr.dstAccessMask       = VK_ACCESS_TRANSFER_READ_BIT;

            vkCmdPipelineBarrier(a_cmdBuff,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    VK_PIPELINE_STAGE_TRANSFER_BIT,
                    0,
                    0, nullptr,
                    1, &bufBarr,
                    0, nullptr);

            VkBufferCopy copyInfo{};
            copyInfo.dstOffset = 0;
            copyInfo.srcOffset = 0;
            copyInfo.size      = a_bufferSize;

            vkCmdCopyBuffer(a_cmdBuff, a_bufferGPU, a_bufferStaging, 1, &copyInfo);

            VK_CHECK_RESULT(vkEndCommandBuffer(a_cmdBuff));
        }


        static void RecordCommandsOfCopyImageDataToTexture(VkCommandBuffer a_cmdBuff, VkPipeline a_pipeline, int a_width, int a_height, VkBuffer a_bufferDynamic,
                VkImage *a_images, VkBuffer a_bufferStaging)
        {
            VkCommandBufferBeginInfo beginInfo{};
            beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            VK_CHECK_RESULT(vkBeginCommandBuffer(a_cmdBuff, &beginInfo));

            vkCmdFillBuffer(a_cmdBuff, a_bufferStaging, 0, a_width * a_height * sizeof(float) * 4, 0);
            // clear this buffer just for an example and test cases. if we comment 'vkCmdCopyBuffer', we'll get black image

            VkImageSubresourceRange rangeWholeImage = WholeImageRange();

            VkImageSubresourceLayers shittylayers{};
            shittylayers.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
            shittylayers.mipLevel       = 0;
            shittylayers.baseArrayLayer = 0;
            shittylayers.layerCount     = 1;

            VkBufferImageCopy wholeRegion = {};
            wholeRegion.bufferOffset      = 0;
            wholeRegion.bufferRowLength   = uint32_t(a_width);
            wholeRegion.bufferImageHeight = uint32_t(a_height);
            wholeRegion.imageExtent       = VkExtent3D{uint32_t(a_width), uint32_t(a_height), 1};
            wholeRegion.imageOffset       = VkOffset3D{0,0,0};
            wholeRegion.imageSubresource  = shittylayers;

            VkImageMemoryBarrier moveToGeneralBar = imBarTransfer(a_images[0],
                    rangeWholeImage,
                    VK_IMAGE_LAYOUT_UNDEFINED,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

            vkCmdPipelineBarrier(a_cmdBuff,
                    VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                    VK_PIPELINE_STAGE_TRANSFER_BIT,
                    0,
                    0, nullptr,            // general memory barriers
                    0, nullptr,            // buffer barriers
                    1, &moveToGeneralBar); // image  barriers

            VkClearColorValue clearVal = {};
            clearVal.float32[0] = 1.0f;
            clearVal.float32[1] = 1.0f;
            clearVal.float32[2] = 1.0f;
            clearVal.float32[3] = 1.0f;

            vkCmdClearColorImage(a_cmdBuff, a_images[0], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clearVal, 1, &rangeWholeImage);

            vkCmdCopyBufferToImage(a_cmdBuff, a_bufferDynamic, a_images[0], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &wholeRegion);

            VkImageMemoryBarrier imgBar = {};
            {
                imgBar.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                imgBar.pNext = nullptr;
                imgBar.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                imgBar.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

                imgBar.srcAccessMask       = 0;
                imgBar.dstAccessMask       = VK_ACCESS_SHADER_READ_BIT;
                imgBar.oldLayout           = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                imgBar.newLayout           = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                imgBar.image               = a_images[0];

                imgBar.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
                imgBar.subresourceRange.baseMipLevel   = 0;
                imgBar.subresourceRange.levelCount     = 1;
                imgBar.subresourceRange.baseArrayLayer = 0;
                imgBar.subresourceRange.layerCount     = 1;
            };

            vkCmdPipelineBarrier(a_cmdBuff,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    0,
                    0, nullptr,
                    0, nullptr,
                    1, &imgBar);

            //VkImageMemoryBarrier barForCopy[2];
            //barForCopy[0] = imBarTransfer(a_images[0], rangeWholeImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
            //barForCopy[1] = imBarTransfer(a_images[1], rangeWholeImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
            //
            //vkCmdPipelineBarrier(a_cmdBuff,
            //                     VK_PIPELINE_STAGE_TRANSFER_BIT,
            //                     VK_PIPELINE_STAGE_TRANSFER_BIT,
            //                     0,
            //                     0, nullptr,     // general memory barriers
            //                     0, nullptr,     // buffer barriers
            //                     2, barForCopy); // image  barriers
            //
            //vkCmdCopyImageToBuffer(a_cmdBuff, a_images[1], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, a_bufferStaging, 1, &wholeRegion);


            VK_CHECK_RESULT(vkEndCommandBuffer(a_cmdBuff)); // end recording commands.
        }

        static void RunCommandBuffer(VkCommandBuffer a_cmdBuff, VkQueue a_queue, VkDevice a_device)
        {
            VkSubmitInfo submitInfo{};
            submitInfo.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers    = &a_cmdBuff;

            VkFence fence{};
            VkFenceCreateInfo fenceCreateInfo{};
            fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            fenceCreateInfo.flags = 0;
            VK_CHECK_RESULT(vkCreateFence(a_device, &fenceCreateInfo, NULL, &fence));
            VK_CHECK_RESULT(vkQueueSubmit(a_queue, 1, &submitInfo, fence));
            VK_CHECK_RESULT(vkWaitForFences(a_device, 1, &fence, VK_TRUE, 100000000000));
            vkDestroyFence(a_device, fence, NULL);
        }

        void Cleanup()
        {
            if (enableValidationLayers)
            {
                auto func = (PFN_vkDestroyDebugReportCallbackEXT)vkGetInstanceProcAddr(m_instance, "vkDestroyDebugReportCallbackEXT");
                if (func == nullptr) {
                    throw std::runtime_error("Could not load vkDestroyDebugReportCallbackEXT");
                }
                func(m_instance, m_debugReportCallback, NULL);
            }

            if(m_bufferDynamic != NULL)
            {
                vkFreeMemory   (m_device, m_bufferMemoryDynamic, NULL);
                vkDestroyBuffer(m_device, m_bufferDynamic, NULL);
            }

            vkFreeMemory   (m_device, m_bufferMemoryStaging, NULL);
            vkDestroyBuffer(m_device, m_bufferStaging, NULL);

            vkFreeMemory   (m_device, m_bufferMemoryGPU, NULL);
            vkDestroyBuffer(m_device, m_bufferGPU, NULL);

            m_img.Release(m_device);

            vkDestroyShaderModule(m_device, m_computeShaderModule, NULL);
            vkDestroyDescriptorPool(m_device, m_descriptorPool, NULL);
            vkDestroyDescriptorSetLayout(m_device, m_descriptorSetLayout, NULL);
            vkDestroyPipelineLayout(m_device, m_pipelineLayout, NULL);
            vkDestroyPipeline(m_device, m_pipeline, NULL);
            vkDestroyCommandPool(m_device, m_commandPool, NULL);
            vkDestroyDevice(m_device, NULL);
            vkDestroyInstance(m_instance, NULL);
        }

        void Run()
        {
            const int deviceId{0};
            std::cout << "init vulkan for device " << deviceId << " ... \n";

            m_instance = vk_utils::CreateInstance(enableValidationLayers, m_enabledLayers);
            if (enableValidationLayers)
            {
                vk_utils::InitDebugReportCallback(m_instance,
                        &debugReportCallbackFn, &m_debugReportCallback);
            }

            m_physicalDevice = vk_utils::FindPhysicalDevice(m_instance, true, deviceId);

            uint32_t queueFamilyIndex{vk_utils::GetComputeQueueFamilyIndex(m_physicalDevice)};
            m_device = vk_utils::CreateLogicalDevice(queueFamilyIndex, m_physicalDevice, m_enabledLayers);
            vkGetDeviceQueue(m_device, queueFamilyIndex, 0, &m_queue);

            size_t bufferSize{sizeof(Pixel) * WIDTH * HEIGHT};
            std::cout << "creating resources ... \n";
            CreateStagingBuffer(  m_device, m_physicalDevice, bufferSize, &m_bufferStaging, &m_bufferMemoryStaging);
            CreateWriteOnlyBuffer(  m_device, m_physicalDevice, bufferSize, &m_bufferGPU, &m_bufferMemoryGPU);

            // test image filter pipeline here ... temp solution

            int w{}, h{};
            auto imageData{LoadBMP("res/Bathroom_LDR_0001.bmp", &w, &h)};

            std::cout << w << " " << h << "\n";
            if (!imageData.size())
            {
                std::cout << "can't load texture 'Bathroom_LDR_0001.png' \n";
                return;
            }

            m_img = CustomVulkanTexture::Create2DTextureRGBA256(m_device, m_physicalDevice, w, h);


            CreateDescriptorSetLayout(m_device, &m_descriptorSetLayout);
            // here we will create a binding of bufferStaging to shader via descriptorSet
            CreateDescriptorSetForOurBufferAndTexture(m_device, m_bufferGPU, bufferSize, &m_descriptorSetLayout, m_img.imageGPU,
                    &m_descriptorPool, &m_descriptorSet, &m_img.imageSampler, &m_img.imageView);
            // (device, bufferGPU, bufferSize, descriptorSetLayout) ==>  #NOTE: we write now to 'bufferGPU', not 'bufferStaging'
            // (descriptorPool, descriptorSet, imageSamplers, imageViews)

            std::cout << "compiling shaders  ... \n";
            CreateComputePipeline(m_device, m_descriptorSetLayout,
                    &m_computeShaderModule, &m_pipeline, &m_pipelineLayout);

            CreateCommandBuffer(m_device, queueFamilyIndex, m_pipeline, m_pipelineLayout, &m_commandPool, &m_commandBuffer);

            // load texture data to GPU
            {
                CreateDynamicBuffer(m_device, m_physicalDevice, w * h * sizeof(int), &m_bufferDynamic, &m_bufferMemoryDynamic);

                // bufferMemoryDynamic <== imageData.data()
                {
                    void *mappedMemory = nullptr;
                    vkMapMemory(m_device, m_bufferMemoryDynamic, 0, w * h * sizeof(int), 0, &mappedMemory);
                    memcpy(mappedMemory, imageData.data(), w * h * sizeof(int));
                    vkUnmapMemory(m_device, m_bufferMemoryDynamic);
                }

                vkResetCommandBuffer(m_commandBuffer, 0);
                RecordCommandsOfCopyImageDataToTexture(m_commandBuffer, m_pipeline, w, h, m_bufferDynamic, // bufferDynamic ==> imageGPU
                        &m_img.imageGPU, m_bufferStaging);

                std::cout << "doing some computations ... \n";
                RunCommandBuffer(m_commandBuffer, m_queue, m_device);
            }

            {
                RecordCommandsOfExecuteAndTransfer(m_commandBuffer, m_pipeline, m_pipelineLayout, m_descriptorSet, m_img.imageGPU,
                        bufferSize, m_bufferGPU, m_bufferStaging);
                std::cout << "doing computations ... \n";
                RunCommandBuffer(m_commandBuffer, m_queue, m_device);
                // The former command rendered a mandelbrot set to a bufferStaging.
                // Save that bufferStaging as a png on disk.
                std::cout << "geting image back  ... \n";
                //saveRenderedImageFromDeviceMemory(device, bufferMemoryStaging, 0, WIDTH, HEIGHT);
                std::vector<uint32_t> resultData( WIDTH * HEIGHT);
                GetImageFromGPU(m_device, m_bufferMemoryStaging, WIDTH, HEIGHT,
                        resultData.data());

                std::cout << "saving image       ... \n";
                SaveBMP("result.bmp", resultData.data(), WIDTH, HEIGHT);
                resultData = std::vector<uint32_t>();
                std::cout << std::endl;
            }

            std::cout << "destroying all     ... \n";
            Cleanup();
        }
};

int main()
{
    ComputeApplication app{};

    try
    {
        app.Run();
    }
    catch (const std::runtime_error& e)
    {
        printf("%s\n", e.what());
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
