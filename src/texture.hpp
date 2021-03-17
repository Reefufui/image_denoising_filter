#ifndef TEXTURE_HPP
#define TEXTURE_HPP

#include "vk_utils.h"

class CustomVulkanTexture
{
    private:
        VkDeviceMemory m_imagesMemoryGPU{}; 
        VkImage        m_imageGPU{};
        VkSampler      m_imageSampler{};
        VkImageView    m_imageView{};
        VkDevice       m_device{};

        bool m_used{};

    public:

        VkDeviceMemory getDeviceMemory() { return m_imagesMemoryGPU; }
        VkImage        getImage()        { return m_imageGPU; }
        VkImage*       getpImage()       { return &m_imageGPU; }
        VkSampler      getSampler()      { return m_imageSampler; }
        VkImageView    getImageView()    { return m_imageView; }

        CustomVulkanTexture()
            : m_imagesMemoryGPU(0), m_imageGPU(0), m_imageSampler(0), m_imageView(0), m_device(0)
        {
        }

        void create(VkDevice a_device, VkPhysicalDevice a_physDevice, const int a_width, const int a_height);
        void release();
};

#endif // TEXTURE_HPP
