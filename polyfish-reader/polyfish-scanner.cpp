#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <set>
#include <map>
#include <fstream>
#include <sstream>
#include <regex>
#include <unistd.h>
#include <sys/uio.h>
#include "reader_util.h"
#include "reader_types.h"

struct MemoryMap {
    uintptr_t start;
    uintptr_t end;
    std::string perms;
    std::string pathname;
};

std::vector<MemoryMap> get_maps(pid_t pid) {
    std::vector<MemoryMap> maps;
    std::string mapsPath = "/proc/" + std::to_string(pid) + "/maps";
    std::ifstream mapsFile(mapsPath);
    std::string line;
    std::regex mapsRegex("([0-9a-f]+)-([0-9a-f]+)\\s+([rwxp-]+)\\s+([0-9a-f]+)\\s+([0-9a-f:]+)\\s+(\\d+)\\s*(.*)");
    
    while (std::getline(mapsFile, line)) {
        std::smatch match;
        if (std::regex_match(line, match, mapsRegex)) {
            MemoryMap m;
            m.start = std::stoull(match[1].str(), nullptr, 16);
            m.end = std::stoull(match[2].str(), nullptr, 16);
            m.perms = match[3].str();
            m.pathname = match[7].str();
            m.pathname.erase(0, m.pathname.find_first_not_of(" \t\r\n"));
            m.pathname.erase(m.pathname.find_last_not_of(" \t\r\n") + 1);
            maps.push_back(m);
        }
    }
    return maps;
}

bool is_valid_addr(uintptr_t addr, const std::vector<MemoryMap>& sorted_maps) {
    auto it = std::lower_bound(sorted_maps.begin(), sorted_maps.end(), addr, [](const MemoryMap& m, uintptr_t val) {
        return m.end <= val;
    });
    return it != sorted_maps.end() && it->start <= addr && it->end > addr;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <pid> [target_turn]" << std::endl;
        return 1;
    }

    pid_t pid = std::stoi(argv[1]);
    int target_turn = (argc > 2) ? std::stoi(argv[2]) : -1;

    std::vector<MemoryMap> all_maps = get_maps(pid);
    std::vector<MemoryMap> sorted_maps = all_maps;
    std::sort(sorted_maps.begin(), sorted_maps.end(), [](const MemoryMap& a, const MemoryMap& b) {
        return a.start < b.start;
    });

    uintptr_t ga_base = 0;
    for (const auto& m : all_maps) {
        if (m.pathname.find("GameAssembly.dll") != std::string::npos) {
            ga_base = m.start;
            std::cout << "GameAssembly.dll base: 0x" << std::hex << ga_base << std::dec << std::endl;
            break;
        }
    }

    if (!ga_base) {
        std::cerr << "GameAssembly.dll not found!" << std::endl;
        return 1;
    }

    std::vector<uintptr_t> instances;
    std::cout << "Finding GameManager instances..." << std::endl;

    for (const auto& m : all_maps) {
        if (m.perms.find("rw") == std::string::npos || (m.end - m.start) > 200 * 1024 * 1024) {
            continue;
        }

        size_t size = m.end - m.start;
        std::vector<unsigned char> data(size);
        if (!readBlock(pid, m.start, data.data(), size)) continue;

        for (size_t i = 0; i < size - 128; i += 8) {
            uintptr_t cb_ptr = *(uintptr_t*)&data[i + 0x28];
            if (cb_ptr == 0 || !is_valid_addr(cb_ptr, sorted_maps)) continue;

            uintptr_t gs_ptr;
            if (!readPiece(pid, cb_ptr + 0x38, gs_ptr) || !is_valid_addr(gs_ptr, sorted_maps)) continue;

            uint32_t turn;
            if (!readPiece(pid, gs_ptr + 0x18, turn)) continue;

            if (target_turn != -1 && (int)turn != target_turn) continue;

            uintptr_t ps_list_ptr;
            if (!readPiece(pid, gs_ptr + 0x38, ps_list_ptr) || !is_valid_addr(ps_list_ptr, sorted_maps)) continue;

            uintptr_t items_ptr;
            if (!readPiece(pid, ps_list_ptr + 0x10, items_ptr) || !is_valid_addr(items_ptr, sorted_maps)) continue;

            uint32_t count;
            if (!readPiece(pid, ps_list_ptr + 0x18, count) || count == 0 || count > 32) continue;

            bool player_match = false;
            for (uint32_t p = 0; p < count; ++p) {
                uintptr_t p_ptr;
                if (!readPiece(pid, items_ptr + 0x20 + p * 8, p_ptr) || !is_valid_addr(p_ptr, sorted_maps)) continue;
                int32_t currency, score;
                if (!readPiece(pid, p_ptr + 0x9C, currency)) continue;
                if (!readPiece(pid, p_ptr + 0xA0, score)) continue;
                if (currency == 1 && score == 1030) {
                    player_match = true;
                    break;
                }
            }
            if (!player_match) continue;

            uintptr_t inst_addr = m.start + i;
            std::cout << "Candidate Match at 0x" << std::hex << inst_addr << " (Turn: " << std::dec << turn << ")" << std::endl;
            instances.push_back(inst_addr);
        }
    }

    if (instances.empty()) {
        std::cout << "No instances found." << std::endl;
        return 0;
    }

    std::cout << "Step 2: Finding pointers to G (Y such that [Y + 0x0] == G)..." << std::endl;
    std::set<uintptr_t> pointers_to_g;
    for (const auto& m : all_maps) {
        if (m.perms.find("rw") == std::string::npos || (m.end - m.start) > 200 * 1024 * 1024) continue;
        size_t size = m.end - m.start;
        std::vector<unsigned char> data(size);
        if (!readBlock(pid, m.start, data.data(), size)) continue;
        for (size_t i = 0; i < size - 8; i += 8) {
            uintptr_t val = *(uintptr_t*)&data[i];
            for (uintptr_t inst : instances) {
                if (val == inst) {
                    pointers_to_g.insert(m.start + i);
                }
            }
        }
    }

    std::cout << "Step 3: Finding class pointers (X such that [X + 0xB8] == Y)..." << std::endl;
    std::set<uintptr_t> class_pointers;
    for (const auto& m : all_maps) {
        if (m.perms.find("rw") == std::string::npos || (m.end - m.start) > 200 * 1024 * 1024) continue;
        size_t size = m.end - m.start;
        std::vector<unsigned char> data(size);
        if (!readBlock(pid, m.start, data.data(), size)) continue;
        for (size_t i = 0; i < size - 0xC0; i += 8) {
            uintptr_t val_at_b8 = *(uintptr_t*)&data[i + 0xB8];
            if (pointers_to_g.count(val_at_b8)) {
                class_pointers.insert(m.start + i);
                std::cout << "Found potential class pointer X at 0x" << std::hex << (m.start + i) << " pointing to Y at 0x" << val_at_b8 << std::endl;
            }
        }
    }

    std::cout << "Step 4: Finding static pointers in GameAssembly.dll pointing to X..." << std::endl;
    for (const auto& m : all_maps) {
        if (m.pathname.find("GameAssembly") == std::string::npos && 
            !(m.start >= ga_base && m.start < ga_base + 0x30000000 && m.pathname == "")) {
            continue;
        }
        if (m.perms.find("r") == std::string::npos) continue;

        size_t size = m.end - m.start;
        std::vector<unsigned char> data(size);
        if (!readBlock(pid, m.start, data.data(), size)) continue;

        for (size_t i = 0; i < size - 8; i += 8) {
            uintptr_t val = *(uintptr_t*)&data[i];
            if (class_pointers.count(val)) {
                uintptr_t addr = m.start + i;
                uintptr_t offset = addr - ga_base;
                std::cout << "\n[!!!] SUCCESS! found static offset: GameAssembly.dll + 0x" << std::hex << offset << std::endl;
            }
        }
    }

    return 0;
}
