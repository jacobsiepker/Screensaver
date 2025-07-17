#pragma once
#include <optional>
#include <cstdint>

// This is where we store all of the queue families that we will want to get from the device
struct QueueFamilyIndices{
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;

    bool isComplete() {
        return graphicsFamily.has_value();
    }
};