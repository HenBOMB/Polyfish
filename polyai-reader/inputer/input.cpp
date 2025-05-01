#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <unistd.h>
#include <sys/uio.h>
#include <cstdint>

uintptr_t getModuleBase(pid_t pid) {
    std::string mapsPath = "/proc/" + std::to_string(pid) + "/maps";
    std::ifstream maps(mapsPath);
    if (!maps.is_open()) return 0;

    std::string line;
    std::string modName = "GameAssembly.dll";
    while (std::getline(maps, line)) {
        if (line.find(modName) != std::string::npos) {
            uintptr_t base = std::stoull(line.substr(0, line.find('-')), nullptr, 16);
            maps.close();
            return base;
        }
    }
    maps.close();
    return 0;
}

bool readBlock(pid_t pid, uintptr_t addr, void* buffer, size_t size) {
    struct iovec local[1];
    struct iovec remote[1];

    local[0].iov_base = buffer;
    local[0].iov_len = size;
    remote[0].iov_base = (void*)addr;
    remote[0].iov_len = size;

    ssize_t nread = process_vm_readv(pid, local, 1, remote, 1, 0);
    return nread == size;
}

template <typename T>
bool readMemory(pid_t pid, uintptr_t addr, T &value) {
    return readBlock(pid, addr, &value, sizeof(T));
}

uintptr_t getPointer(pid_t pid, uintptr_t base, const std::vector<uintptr_t>& offsets) {
    uintptr_t addr = base;
    for (size_t i = 0; i < offsets.size(); ++i) {
        if (!readMemory(pid, addr, addr)) {
            return 0;
        }
        addr += offsets[i];
    }
    return addr;
}

// uintptr_t selectedTileAddress = getPointer(pid, baseAddress, { 0xB8, 0x10, 0x0 });

int main(int argc, char** argv) {
    bool prod = argc > 2 && argv[2] != nullptr && std::string(argv[2]) == "-y"? true : false;
    pid_t pid = std::stoi(argv[1]);

    uintptr_t baseAddress = getModuleBase(pid);

    if (!baseAddress) {
        std::cerr << "Failed to get module base address.\n";
        return 1;
    }

    // ActionManager (63b0e8a0) 
    //   Constructor: (gameState: GameState) (GameState is huge datatype)
    //   Methods: ExecuteCommand(command: CommandBase, error: Ssytem.String&)

    // CommandBase (63727f60, inherits System.Object)
    //   Fields: 0x10 - playerId (byte)
    //   Constructor: (playerId: Byte)
        
    // ResearchCommand (64052060, inherits CommandBase)
    //   Fields: 0x18 - TechType

    // ResearchCommand
    uintptr_t researchCommandAddress = 0x640520a0;

    // I want to create an instance of that research command object

    return 0;
}
