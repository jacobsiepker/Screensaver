#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <stdexcept>
#include <vector>
#include <cstdlib>
#include <cstdint>
#include <algorithm>
#include <QueueFamilyIndices.h>

const uint32_t WIDTH = 1920;
const uint32_t HEIGHT = 1080;

const int MAX_FRAMES_IN_FLIGHT = 2;

class ScreensaverWindow{

public:
    GLFWwindow* window;
    bool closeProgramRequest = false;

private:
    bool framebufferResized = false;
    VkSurfaceKHR surface; 
    
    // The Vk Instance and logical device are common across all windows
    VkInstance* pVkInstance;
    VkPhysicalDevice* pVkPhysicalDevice;
    VkDevice* pVkDevice;


public:

    ScreensaverWindow(VkInstance* _pVkInstance, VkPhysicalDevice* _pVkPhysicalDevice, VkDevice* _pVkDevice, QueueFamilyIndices)
        : pVkInstance(_pVkInstance), pVkPhysicalDevice(_pVkPhysicalDevice), pVkDevice(_pVkDevice)
    {
    }

    void initWindow()
    {
        glfwInit(); // Prep GLFW Window
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API); // Dont load OpenGL context
        glfwWindowHint(GLFW_TRANSPARENT_FRAMEBUFFER, GLFW_TRUE);
        glfwWindowHint(GLFW_DECORATED, GLFW_FALSE);
        //glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE); // Window is not resizeable
        window = glfwCreateWindow(WIDTH, HEIGHT, "Screensaver Demo", nullptr, nullptr);
        glfwSetWindowUserPointer(window, this);
        glfwSetFramebufferSizeCallback(window, framebufferResizeCallback); // set a callback for window resize events
        glfwSetKeyCallback(window, keyCallback);
        //glfwSetWindowOpacity(window, 0.5); // window opacity example
    }

    static void framebufferResizeCallback(GLFWwindow* window, int width, int height) {
        auto app = reinterpret_cast<ScreensaverWindow*>(glfwGetWindowUserPointer(window));
        app->framebufferResized = true;
    }

    static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
        if (action == GLFW_PRESS) {
            auto app = reinterpret_cast<ScreensaverWindow*>(glfwGetWindowUserPointer(window));
            app->closeProgramRequest = true;
        }
    }


    void createSurface() {
        if (glfwCreateWindowSurface(*pVkInstance, window, nullptr, &surface) != VK_SUCCESS) {
            throw std::runtime_error("Womp Womp! Failed to create window surface.");
        }
    }
 
    void recreateSwapChain() {

        // Host will get caught here during minimization, causing no rendering to happen.
        int width = 0, height = 0;
        glfwGetFramebufferSize(window, &width, &height);
        while (width == 0 || height == 0) {
            glfwGetFramebufferSize(window, &width, &height);
            glfwWaitEvents();
        }

        vkDeviceWaitIdle(*pVkDevice); // no swap chain recreation during rendering

        cleanupSwapChain();

        createSwapChain();
        createImageViews(); // creation depends on swap chain params
        createFrameBuffers(); // creation depends on swap chain params
    }

    struct SwapChainSupportDetails { // Because why would just any swap chain work? That would be too easy.
        VkSurfaceCapabilitiesKHR capabilities;
        std::vector<VkSurfaceFormatKHR> formats;
        std::vector<VkPresentModeKHR> presentModes;
    };

    SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice _device) {
        SwapChainSupportDetails details;

        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(_device, surface, &details.capabilities);

        // IRS pattern to get surface format information and present mode information
        uint32_t formatCount;
        vkGetPhysicalDeviceSurfaceFormatsKHR(_device, surface, &formatCount, nullptr);
        if (formatCount != 0) {
            details.formats.resize(formatCount);
            vkGetPhysicalDeviceSurfaceFormatsKHR(_device, surface, &formatCount, details.formats.data());
        }

        uint32_t presentModeCount;
        vkGetPhysicalDeviceSurfacePresentModesKHR(_device, surface, &presentModeCount, nullptr);
        if (presentModeCount != 0) {
            details.presentModes.resize(presentModeCount);
            vkGetPhysicalDeviceSurfacePresentModesKHR(_device, surface, &presentModeCount, details.presentModes.data());
        }

        return details;
    }

    bool checkDeviceSwapChainSupport(VkPhysicalDevice _device)
    {
        // Ensure the device has a supported image format and present mode.
        SwapChainSupportDetails swapChainSupport = querySwapChainSupport(_device);
        return !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
    }

    VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) {
        for (const auto& availableFormat : availableFormats) {
            if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                return availableFormat;
            }
        }
        return availableFormats[0]; // None of them have the right color space? Fuck it. Just take a format and go. 
    }

    VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) {
        for (const auto& availablePresentMode : availablePresentModes) {
            // Mailbox mode is good for low latency, no tearing, but high power usage and frame skipping.
            if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
                return availablePresentMode;
            }
        }
        return VK_PRESENT_MODE_FIFO_KHR; // FIFO is guaranteed to exist
    }

    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) {
        //Get the size of the display from GLFW
        if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
            return capabilities.currentExtent;
        }
        else {
            int width, height;
            glfwGetFramebufferSize(window, &width, &height);

            VkExtent2D actualExtent = {
                static_cast<uint32_t>(width),
                static_cast<uint32_t>(height)
            };

            actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
            actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

            return actualExtent;
        }
    }

    void createSwapChain() {
        SwapChainSupportDetails swapChainSupport = querySwapChainSupport(*pVkPhysicalDevice);

        VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
        VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
        VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

        uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
        // ensure we dont request too many swap chain images. (maxImageCount of 0 means no limit)
        if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount) {
            imageCount = swapChainSupport.capabilities.maxImageCount;
        }

        // Start struct filling
        VkSwapchainCreateInfoKHR createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        createInfo.surface = surface;
        createInfo.minImageCount = imageCount;
        createInfo.imageFormat = surfaceFormat.format;
        createInfo.imageColorSpace = surfaceFormat.colorSpace;
        createInfo.imageExtent = extent;
        createInfo.imageArrayLayers = 1;
        createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT; //VK_IMAGE_USAGE_TRANSFER_DST_BIT to render offscreen first

        QueueFamilyIndices indices = findQueueFamilies(pVkPhysicalDevice);
        uint32_t queueFamilyIndices[] = { indices.graphicsFamily.value(), indices.presentFamily.value() };

        //if (indices.graphicsFamily != indices.presentFamily) { // this will never be the case with our current setup
        //    createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        //    createInfo.queueFamilyIndexCount = 2;
        //    createInfo.pQueueFamilyIndices = queueFamilyIndices;
        //}
        //else {
        createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        createInfo.queueFamilyIndexCount = 0; // Optional
        createInfo.pQueueFamilyIndices = nullptr; // Optional
        //}
        createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
        createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR;// VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR; // You could use this to blend over other applications in windows system. Pretty cool Imo.
        createInfo.presentMode = presentMode;
        createInfo.clipped = VK_TRUE; // Disable this to enable the rendering behind the Vk application, for semi-transparent effects.
        createInfo.oldSwapchain = VK_NULL_HANDLE; // Necessary for resizing, since swap chains will need to be discarded and recreated.


        // Try to create the swapchain
        if (vkCreateSwapchainKHR(*pVkDevice, &createInfo, nullptr, &swapChain) != VK_SUCCESS) {
            throw std::runtime_error("Womp Womp! Failed to create swap chain.");
        }

        // IRS pattern to grab pointers into swap chain
        vkGetSwapchainImagesKHR(*pVkDevice, swapChain, &imageCount, nullptr);
        swapChainImages.resize(imageCount);
        vkGetSwapchainImagesKHR(*pVkDevice, swapChain, &imageCount, swapChainImages.data());

        swapChainImageFormat = surfaceFormat.format;
        swapChainExtent = extent;
    }

    void cleanupSwapChain() {
        for (auto framebuffer : swapChainFramebuffers) { vkDestroyFramebuffer(*pVkDevice, framebuffer, nullptr); }
        for (auto imageView : swapChainImageViews) { vkDestroyImageView(*pVkDevice, imageView, nullptr); }
        vkDestroySwapchainKHR(*pVkDevice, swapChain, nullptr);
    }

    void createImageViews() {
        swapChainImageViews.resize(swapChainImages.size());

        // Set basic info for all the swap chain images.
        for (size_t i = 0; i < swapChainImages.size(); i++) {
            VkImageViewCreateInfo createInfo{};
            createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            createInfo.image = swapChainImages[i];
            createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            createInfo.format = swapChainImageFormat;

            createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;

            createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            createInfo.subresourceRange.baseMipLevel = 0;
            createInfo.subresourceRange.levelCount = 1;
            createInfo.subresourceRange.baseArrayLayer = 0;
            createInfo.subresourceRange.layerCount = 1;

            // try to create our image view
            if (vkCreateImageView(*pVkDevice, &createInfo, nullptr, &swapChainImageViews[i]) != VK_SUCCESS) {
                throw std::runtime_error("failed to create image views!");
            }
        }
    }

    void createFrameBuffers() {
        swapChainFramebuffers.resize(swapChainImageViews.size());

        for (size_t i = 0; i < swapChainImageViews.size(); i++) {
            VkImageView attachments[] = {
                swapChainImageViews[i]
            };

            VkFramebufferCreateInfo framebufferInfo{};
            framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebufferInfo.renderPass = renderPass;
            framebufferInfo.attachmentCount = 1;
            framebufferInfo.pAttachments = attachments;
            framebufferInfo.width = swapChainExtent.width;
            framebufferInfo.height = swapChainExtent.height;
            framebufferInfo.layers = 1;

            // try to create the frame buffer
            if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapChainFramebuffers[i]) != VK_SUCCESS) {
                throw std::runtime_error("Womp Womp! Failed to create framebuffer.");
            }
        }
    }

    void createRenderPass() {
        VkAttachmentDescription colorAttachment{};
        colorAttachment.format = swapChainImageFormat; // create color attachment to render from swap chain
        colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT; // multisampling requires different value
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; // clear framebuffer to black before writing frame
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE; // store the frame so we can grab it and see it
        colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE; //  we dont use stencil data, so dont dictate any data policy for it
        colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED; // we would care about prev layout if we didn't clear the frame buffer or were doing multi pass rendering
        colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR; // specify image usage here. PRESENT_SRC_KHR is for images presented in swap chain memory

        // Setup a subpass - each render pass is made up of one or more subpasses.
        VkAttachmentReference colorAttachmentRef{};
        colorAttachmentRef.attachment = 0;
        colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkSubpassDependency dependency{ };
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass = 0;
        dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.srcAccessMask = 0;

        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorAttachmentRef;

        // Create the actual render pass
        // Struct filling
        VkRenderPassCreateInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInfo.attachmentCount = 1;
        renderPassInfo.pAttachments = &colorAttachment;
        renderPassInfo.subpassCount = 1;
        renderPassInfo.pSubpasses = &subpass;
        renderPassInfo.dependencyCount = 1;
        renderPassInfo.pDependencies = &dependency;

        // try to create render pass
        if (vkCreateRenderPass(*pVkDevice, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
            throw std::runtime_error("Womp Womp! Failed to create render pass.");
        }
    }


    void createCommandPool() {
        QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);
        // struct filling
        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();

        //try to create cmd pool
        if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
            throw std::runtime_error("Womp Womp! Failed to create command pool.");
        }
    }

    void createCommandBuffers() {

        commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);

        // very simple vulkanese on this one
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = commandPool; // the guy we just initilized in the method above
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; // if this were secondary, it would be like a cmd buffer group that a primary buffer could call for execution at once
        allocInfo.commandBufferCount = (uint32_t)commandBuffers.size();

        //try to create the cmd buffer
        if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()) != VK_SUCCESS) {
            throw std::runtime_error("Womp Womp! Failed to allocate command buffers.");
        }
    }

    void recordCommandBuffer(VkCommandBuffer _cmdBuffer, uint32_t _imgIndex) {
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = 0;
        beginInfo.pInheritanceInfo = nullptr;

        // start recording buffer
        if (vkBeginCommandBuffer(_cmdBuffer, &beginInfo) != VK_SUCCESS) {
            throw std::runtime_error("Womp Womp! Failed to begin recording command buffer.");
        }

        // Start render pass
        VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = renderPass;
        renderPassInfo.framebuffer = swapChainFramebuffers[_imgIndex];
        renderPassInfo.renderArea.offset = { 0, 0 };
        renderPassInfo.renderArea.extent = swapChainExtent;
        VkClearValue clearColor = { {{0.0f, 0.0f, 0.0f, 0.0f}} };
        renderPassInfo.clearValueCount = 1;
        renderPassInfo.pClearValues = &clearColor; // what color does it clear to before each render pass?

        vkCmdBeginRenderPass(_cmdBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE); // CONTENTS_INLINE = Not using secondary cmd buffers

        // Bind the graphics pipeline
        vkCmdBindPipeline(_cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

        // Specify viewport and scissor info for cmd buffer
        VkViewport viewport{};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = static_cast<float>(swapChainExtent.width);
        viewport.height = static_cast<float>(swapChainExtent.height);
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        vkCmdSetViewport(_cmdBuffer, 0, 1, &viewport);

        VkRect2D scissor{};
        scissor.offset = { 0,0 };
        scissor.extent = swapChainExtent;
        vkCmdSetScissor(_cmdBuffer, 0, 1, &scissor);

        // Set the time as a push constant
        int timeElapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startTime).count();
        vkCmdPushConstants(_cmdBuffer, pipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(int), &timeElapsed);

        // Draw the triangle! (command created, not yet sent or received or queued or run)
        vkCmdDraw(_cmdBuffer, 4, 1, 0, 0);

        vkCmdEndRenderPass(_cmdBuffer);

        if (vkEndCommandBuffer(_cmdBuffer) != VK_SUCCESS) {
            throw std::runtime_error("Womp Womp! Failed to record command buffers");
        }
    }

    void createSyncObjects() {

        imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

        VkSemaphoreCreateInfo semaphoreInfo{};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT; // have it start as signaled so the first frame doesn't wait on something that will never happen.

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
        {
            // try to create both semaphores and the fence in one big statement
            if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS ||
                vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS ||
                vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS) {
                throw std::runtime_error("Womp Womp! Failed to create semaphores.");
            }
        }
    }


    void getInput() {
        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
            closeProgramRequest = true;
        }
    }

};