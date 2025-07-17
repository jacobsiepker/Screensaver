//#include <vulkan/vulkan.h>
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <vector>
#include <optional>
#include <set>
#include <cstdint>
#include <limits>
#include <fstream>
#include <chrono>


const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

const std::vector<const char*> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif


class VulkanScreensaverApplication {


private:
    VkInstance instance;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device;
    VkQueue graphicsQueue;
    VkQueue presentQueue;
    VkSwapchainKHR swapChain;
    std::vector<VkImage> swapChainImages;
    VkFormat swapChainImageFormat;
    VkExtent2D swapChainExtent;
    std::vector<VkImageView> swapChainImageViews;
    VkRenderPass renderPass;
    VkPipelineLayout pipelineLayout;
    VkPipeline graphicsPipeline;
    std::vector<VkFramebuffer> swapChainFramebuffers;
    VkCommandPool commandPool;
    std::vector<VkCommandBuffer> commandBuffers;
    std::vector<VkSemaphore> imageAvailableSemaphores;
    std::vector<VkSemaphore> renderFinishedSemaphores;
    std::vector<VkFence> inFlightFences;
    QueueFamilyIndices queueFamIndices;

    bool closeProgram = false;
    bool framebufferResized = false;
    uint32_t currentFrame = 0;
    std::chrono::steady_clock::time_point startTime;



public:
    void run() {
        startTime = std::chrono::steady_clock::now();

        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }

private:

    void initVulkan() {
        createInstance();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        createSwapChain();
        createImageViews();
        createRenderPass();
        createGraphicsPipeline();
        createFrameBuffers();
        createCommandPool();
        createCommandBuffers();
        createSyncObjects();
    }

    void createInstance()
    {
        // Check validation layer support
        if (enableValidationLayers && !checkValidationLayerSupport()) {
            throw std::runtime_error("Womp Womp! Validation layers requested, but not available.");
        }

        // Fill VkAppInfo -  mostly optional iirc.
        VkApplicationInfo appInfo{};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "First Triangle OwO";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "No Engine";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_4;

        // Fill VkCreateInfo, some details provided by glfw. Def not optional.
        // Add in validation layers here if we want them (spoiler, we do)
        VkInstanceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;

        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions;

        glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

        createInfo.enabledExtensionCount = glfwExtensionCount;
        createInfo.ppEnabledExtensionNames = glfwExtensions;

        if (enableValidationLayers) {
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();
        }
        else {
            createInfo.enabledLayerCount = 0;
        }

        // Query and fill extentsions before initializing
        //  Calls the same function twice, because we just get the count on the first pass
        //  This is necessary to create the vector size.
        uint32_t extensionCount = 0;
        vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);
        std::vector<VkExtensionProperties> extensions(extensionCount);
        vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, extensions.data());

        std::cout << "available extensions:\n";
        for (const auto& extension : extensions) {
            std::cout << '\t' << extension.extensionName << '\n'; // this is why we should burn c++ with fire.
        }

        // We have everything we need to create the Vk instance.
        if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create VkInstance.");
        }
    }

    bool checkValidationLayerSupport()
    {
        // I have a feeling there will be a lot of this syntax
        // I hate this with a burning passion. Like, just give me the damn list!
        uint32_t layerCount;
        vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
        std::vector<VkLayerProperties> availableLayers(layerCount);
        vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());
        
        // Ensure we have all the named layers.
        for (const char* layerName : validationLayers)
        {
            bool layerFound = false;

            for (const auto& layerProperties : availableLayers)
            {
                if (strcmp(layerName, layerProperties.layerName) == 0) {
                    layerFound = true;
                    break;
                }
            }
            if (!layerFound) return false;
        }
        return true;
    }

    void pickPhysicalDevice() {
        
        // Get all the vulkan devices with IRS syntax.
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
        if (deviceCount == 0)
        {
            throw std::runtime_error("Womp Womp! Failed to find any GPU with Vulkan support.");
        }
        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

        // Ensure one of the devices is suitable
        for (const auto& device : devices) {
            if (isDeviceSuitable(device)) { // Set our device to the first availabe, no other preferences checked.
                physicalDevice = device;
                break;
            }
        }
        if (physicalDevice == VK_NULL_HANDLE) {
            throw std::runtime_error("Womp Womp! Failed to find a suitable VkDevice.");
        }
    }

    bool isDeviceSuitable(VkPhysicalDevice _device) {
        // Fetch basic device info like Vk version
        VkPhysicalDeviceProperties deviceProperties;
        vkGetPhysicalDeviceProperties(_device, &deviceProperties);

        // Fetch optional feature support
        VkPhysicalDeviceFeatures deviceFeatures;
        vkGetPhysicalDeviceFeatures(_device, &deviceFeatures);

        // Get queue families on the device
        QueueFamilyIndices queueFamIndicies = findQueueFamilies(_device);

        ////////////////////////////////////////////
        // Make GPU suitablity decisions here
        if (!queueFamIndicies.graphicsFamily.has_value())  return false;
        if (!checkDeviceExtensionSupport(_device)) return false;
        if (!checkDeviceSwapChainSupport(_device)) return false;
        ////////////////////////////////////////////

        return true;
    }

    bool checkDeviceExtensionSupport(VkPhysicalDevice _device) {
        // IRS pattern to get extension list
        uint32_t extensionCount;
        vkEnumerateDeviceExtensionProperties(_device, nullptr, &extensionCount, nullptr);
        std::vector<VkExtensionProperties> availableExtensions(extensionCount);
        vkEnumerateDeviceExtensionProperties(_device, nullptr, &extensionCount, availableExtensions.data());

        // Check against required extensions
        std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());
        for (const auto& extension : availableExtensions) {
            requiredExtensions.erase(extension.extensionName);
        }
        return requiredExtensions.empty();
    }

    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice _device) {
        QueueFamilyIndices indices;
      
        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(_device, &queueFamilyCount, nullptr);

        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(_device, &queueFamilyCount, queueFamilies.data());

        int i = 0;
        for (const auto& queueFamily : queueFamilies) {

            if (indices.isComplete()) { // Early out, we already have all the queues necessary.
                break; 
            }

            // Good for graphics?
            VkBool32 graphicsSupport = false;
            if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                graphicsSupport = true;
            }
            // Good for presenting?
            VkBool32 presentSupport = false;
            vkGetPhysicalDeviceSurfaceSupportKHR(_device, i, surface, &presentSupport);
            
            // Only use queues that can graphic and present. It works for less GPU's, but is more performant to force them on the same queue.
            if (presentSupport && graphicsSupport)
            {
                indices.graphicsFamily = i;
                indices.presentFamily = i;
            }

            i++;
        }
 
        return indices;
    }

    void createLogicalDevice() {
        // Get the queue families we like
        queueFamIndices = findQueueFamilies(physicalDevice);

        // Create queue info
        std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
        std::set<uint32_t> uniqueQueueFamilies = { queueFamIndices.graphicsFamily.value(), queueFamIndices.presentFamily.value() };

        float queuePriority = 1.0f;

        // Struct filling for queue information
        for (uint32_t queueFamily : uniqueQueueFamilies)
        {
            VkDeviceQueueCreateInfo queueCreateInfo{};
            queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO; //wHo aM i??
            queueCreateInfo.queueFamilyIndex = queueFamily; // Fill it with the queue info we queried from the physicalDevice
            queueCreateInfo.queueCount = 1;
            queueCreateInfo.pQueuePriorities = &queuePriority;
            queueCreateInfos.push_back(queueCreateInfo);
        }

        // Fill out the device features that we will need. None for now
        VkPhysicalDeviceFeatures deviceFeatures{};

        // Bring togeather the stucts above to create the logical device
        VkDeviceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        createInfo.pQueueCreateInfos = queueCreateInfos.data();
        createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
        createInfo.pEnabledFeatures = &deviceFeatures;
        createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
        createInfo.ppEnabledExtensionNames = deviceExtensions.data();

        if (enableValidationLayers) { // Enable validation layers to be compatable with older versions. No longer necessary with new hardware.
            createInfo.enabledLayerCount = static_cast<uint32_t> (validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();
        }
        else {
            createInfo.enabledLayerCount = 0;
        }

        // Try the device creation
        if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS)
        {
            throw std::runtime_error("Womp Womp! Failed to create Vulkan logical device.");
        }

        // The vkCreateDevice did a bunch of queue setup, now we just need to grab the handle to the queues and save them
        vkGetDeviceQueue(device, queueFamIndices.graphicsFamily.value(), 0, &graphicsQueue);
        vkGetDeviceQueue(device, queueFamIndices.presentFamily.value(), 0, &presentQueue);
    }

    void createGraphicsPipeline() {
        auto vertShaderCode = readFile("vert2.spv");
        auto fragShaderCode = readFile("frag2.spv");

        std::cout << "Vert Shader Size : " << vertShaderCode.size() << '\n';
        std::cout << "Frag Shader Size : " << fragShaderCode.size() << '\n';

        // Turn SPIR.V byte code into a VkStructceptionObject
        VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
        VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

        VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
        vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT; // what stage is this shader for?
        vertShaderStageInfo.module = vertShaderModule; // where is the code?
        vertShaderStageInfo.pName = "main";

        VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
        fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        fragShaderStageInfo.module = fragShaderModule;
        fragShaderStageInfo.pName = "main";

        VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

        // Details about incoming vertex data. There is no incoming vertex data at the moment.
        VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInputInfo.vertexBindingDescriptionCount = 0;
        vertexInputInfo.pVertexBindingDescriptions = nullptr;
        vertexInputInfo.vertexAttributeDescriptionCount = 0;
        vertexInputInfo.pVertexAttributeDescriptions = nullptr;

        // How does this pipeline read verticies?
        VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN;//VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        inputAssembly.primitiveRestartEnable = VK_FALSE;

        // Set viewport details for the pipeline
        VkViewport viewport{};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = (float)swapChainExtent.width; // swapchain size is not necessarily the same as monitor size, but it is what we need here.
        viewport.height = (float)swapChainExtent.height;
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;

        // Set scissor details for the pipeline
        // We want nothing to be scissored off
        VkRect2D scissor{};
        scissor.offset = { 0, 0 };
        scissor.extent = swapChainExtent;

        // Enable dynamic viewport/scissoring for widow size changing on the fly if we want it. No penalty on performance if we dont.
        std::vector<VkDynamicState> dynamicStates = {
            VK_DYNAMIC_STATE_VIEWPORT,
            VK_DYNAMIC_STATE_SCISSOR
        };

        VkPipelineDynamicStateCreateInfo dynamicState{};
        dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
        dynamicState.pDynamicStates = dynamicStates.data();

        VkPipelineViewportStateCreateInfo viewportState{};
        viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = 1;
        viewportState.scissorCount = 1;
        // If we did not want dynamic viewports, then we can remove the dynamicState stuff above and include the two following lines.
        //viewportState.pViewports = &viewport;
        //viewportState.pScissors = &scissor;

        // Setup rasterizer fixed function pipeline
        // Lots of fun params here!
        VkPipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.depthClampEnable = VK_FALSE; // For geometry outside of the clippling plane, do we cull or clamp? // Clamp requires extra GPU features
        rasterizer.rasterizerDiscardEnable = VK_FALSE; // VK_TRUE will bypass the rasterizer stage
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL; // How do we draw verts? Fill, line, or point. // Non-fill requires GPU features
        rasterizer.lineWidth = 1.0f; // greater than 1 requires GPU features
        rasterizer.cullMode = VK_CULL_MODE_BACK_BIT; // front/back/no face culling.
        rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE; // vertex ordering. This is the fun one that will mess with your geometry in unique ways.
        
        // You can do some manual depth fuckery here. This could be very fun to play with later.
        rasterizer.depthBiasEnable = VK_FALSE;
        rasterizer.depthBiasConstantFactor = 0.0f;
        rasterizer.depthBiasClamp = 0.0f;
        rasterizer.depthBiasSlopeFactor = 0.0f;


        // Setup multisampling anti-aliasing
        // Off for now - requires additonal GPU features
        VkPipelineMultisampleStateCreateInfo multisampling{};
        multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.sampleShadingEnable = VK_FALSE;
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        multisampling.minSampleShading = 1.0f;
        multisampling.pSampleMask = nullptr;
        multisampling.alphaToCoverageEnable = VK_FALSE;
        multisampling.alphaToOneEnable = VK_FALSE;

        // Depth + Stencil
        // Configure Depth and stencil tests here, if we want to include them.

        // Configure Color Blending. Combines frag shader color with color in framebuffer
        // If we want alpha transparency, we will have to change some params here
        VkPipelineColorBlendAttachmentState colorBlendAttachment{};
        colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachment.blendEnable = VK_FALSE; // Ignores color blending step if set to VK_FALSE
        colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
        colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD; // different color blending function here
        colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;

        VkPipelineColorBlendStateCreateInfo colorBlending{};
        colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlending.logicOpEnable = VK_FALSE; // Enable to use bitwise blending method. Not quite sure what that entails.
        colorBlending.logicOp = VK_LOGIC_OP_COPY;
        colorBlending.attachmentCount = 1;
        colorBlending.pAttachments = &colorBlendAttachment;
        colorBlending.blendConstants[0] = 0.0f;
        colorBlending.blendConstants[1] = 0.0f;
        colorBlending.blendConstants[2] = 0.0f;
        colorBlending.blendConstants[3] = 0.0f;

        // Configure pipeline layout here. None for now
        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 0;
        pipelineLayoutInfo.pSetLayouts = nullptr;
        pipelineLayoutInfo.pushConstantRangeCount = 0;
        pipelineLayoutInfo.pPushConstantRanges = nullptr;

        // try to create the pipeline layout
        if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
            throw std::runtime_error("Womp Womp! Failed to create pipeline layout.");
        }

        //Struct filling for the actual pipeline to be created
        //Most of the other structs are funneling into this one
        VkGraphicsPipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo.stageCount = 2;
        pipelineInfo.pStages = shaderStages; // ref to Vk SPIR-V data
        pipelineInfo.pVertexInputState = &vertexInputInfo;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState = &multisampling;
        pipelineInfo.pDepthStencilState = nullptr;
        pipelineInfo.pColorBlendState = &colorBlending;
        pipelineInfo.pDynamicState = &dynamicState;
        pipelineInfo.layout = pipelineLayout; // vk handle this time, not a ptr
        pipelineInfo.renderPass = renderPass;
        pipelineInfo.subpass = 0;
        pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
        pipelineInfo.basePipelineIndex = -1;

        //try to create the pipeline
        if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS){ // can be used to create multiple pipelines at once!
            throw std::runtime_error("Womp Womp! Failed to create graphics pipeline.");
        }

        // clean up shader modules after pipeline is created. Pipeline has its own copy now.
        vkDestroyShaderModule(device, fragShaderModule, nullptr);
        vkDestroyShaderModule(device, vertShaderModule, nullptr);
    }

    static std::vector<char> readFile(const std::string& filename) {
        std::ifstream file(filename, std::ios::ate | std::ios::binary);

        if (!file.is_open()) {
            throw std::runtime_error("Womp Womp! Failed to open file.");
        }

        size_t fileSize = (size_t)file.tellg();
        std::vector<char> buffer(fileSize);

        file.seekg(0);
        file.read(buffer.data(), fileSize);

        file.close();

        return buffer;
    }

    VkShaderModule createShaderModule(const std::vector<char>& code) {
        // struct filling. Spir-v byte code to VkStructceptionObject
        VkShaderModuleCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.codeSize = code.size();
        createInfo.pCode = reinterpret_cast<const uint32_t*> (code.data()); // we have to case from char* to uint32_t*
        
        // try to create the shader module from VkStructceptionObject
        VkShaderModule shaderModule;
        if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
            throw std::runtime_error("Womp Womp! Failed to create shader module.");
        }

        return shaderModule;
    }


    //////// Runtime

    void drawFrame() {

        vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

        // Get the image from the swapchain
        uint32_t imgIndex;
        VkResult result = vkAcquireNextImageKHR(device, swapChain, UINT64_MAX, imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imgIndex);

        if (result == VK_ERROR_OUT_OF_DATE_KHR) { // window has most likely been resized
            recreateSwapChain();
            return; // dont render at this point, whatever is in the framebuffer is old
        }
        else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
            throw std::runtime_error("Womp Womp! Failed to acquire swap chain image.");
        }

        vkResetFences(device, 1, &inFlightFences[currentFrame]);

        // Stop the old lists. Send out the new lists.
        vkResetCommandBuffer(commandBuffers[currentFrame], 0);
        recordCommandBuffer(commandBuffers[currentFrame], imgIndex);

        // Submit commands to the queue
        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        VkSemaphore waitSemaphores[] = { imageAvailableSemaphores[currentFrame] }; // what semaphores do we wait on to do this command?
        VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = waitSemaphores; 
        submitInfo.pWaitDstStageMask = waitStages;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffers[currentFrame]; // what commands are we submitting?
        VkSemaphore signalSemaphores[] = { renderFinishedSemaphores[currentFrame] }; // what semaphores do we flip after this cmd is done?
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = signalSemaphores;

        if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]) != VK_SUCCESS) {
            throw std::runtime_error("Womp Womp! Failed to submit draw command buffer.");
        }

        // gather the info the present
        VkPresentInfoKHR presentInfo{};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = signalSemaphores;
        VkSwapchainKHR swapChains[] = { swapChain };
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = swapChains;
        presentInfo.pImageIndices = &imgIndex;
        presentInfo.pResults = nullptr; // holds result of last present call, useful if using multiple swap chains since we would lose access to the result

        result = vkQueuePresentKHR(presentQueue, &presentInfo);

        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized) {
            framebufferResized = false;
            recreateSwapChain();
        }
        else if (result != VK_SUCCESS) {
            throw std::runtime_error("Womp Womp! Failed to present swap chain image.");
        }

        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    void mainLoop() {
        while (!closeProgram && !glfwWindowShouldClose(window))
        {
            glfwPollEvents();
            getInput();
            drawFrame();
        }
        vkDeviceWaitIdle(device); // dont destroy anything until we are done with our current rendering commands. shouldn't be long.
    }

    //////// End Runtime

    void cleanup()
    {
        cleanupSwapChain();
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++){
            vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
            vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);
            vkDestroyFence(device, inFlightFences[i], nullptr);}
        vkDestroyCommandPool(device, commandPool, nullptr);
        vkDestroyPipeline(device, graphicsPipeline, nullptr);
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        vkDestroyRenderPass(device, renderPass, nullptr);
        vkDestroyDevice(device, nullptr);
        vkDestroySurfaceKHR(instance, surface, nullptr);
        vkDestroyInstance(instance, nullptr);
        glfwDestroyWindow(window);
        glfwTerminate();
    }
};

int WinMain() {
    VulkanScreensaverApplication app;

    try {
        app.run();
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}