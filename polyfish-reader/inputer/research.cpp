#include <iostream>
#include <cstdint>

typedef bool (*ExecuteCommandFn)(void*, void*, void**);

struct ResearchCommand {
    uintptr_t vtable;    // 0x0
    uint8_t padding[8];  // 0x8-0xF
    uint8_t playerId;    // 0x10
    uint8_t padding2[3]; // 0x11-0x13
    int32_t techId;      // 0x18
};

extern "C" void run_command(int32_t techId) {
    void* actionManager = reinterpret_cast<void*>(0x693F3880);
    ExecuteCommandFn executeCommand = reinterpret_cast<ExecuteCommandFn>(0x64004F28);

    ResearchCommand cmd = {};
    cmd.vtable = 0x6404fff0;  // ResearchCommand vtable
    cmd.playerId = 1;         // Player 1
    cmd.techId = techId;      // Custom techId

    void* errorString = nullptr;

    bool result = executeCommand(actionManager, &cmd, &errorString);

    // Debug output
    std::cout << "Result: " << result << "\n";
    if (!result && errorString) {
        std::cout << "Error: " << reinterpret_cast<char*>(errorString) << "\n";
    }
}

int main() {
    run_command(42);  // Test with techId 42
    return 0;
}