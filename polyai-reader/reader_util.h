#ifndef READER_UTIL_H
#define READER_UTIL_H

#include "reader_types.h"
#include <vector>
#include <sys/uio.h>
#include <cstdint>

uintptr_t getPlace(pid_t pid, uintptr_t base, const std::vector<uintptr_t>& offsets);

bool readBlock(pid_t pid, uintptr_t addr, void* buffer, size_t size);

template <typename T>
bool readPiece(pid_t pid, uintptr_t addr, T &result) {
    return readBlock(pid, addr, &result, sizeof(T));
}

bool readWord(pid_t pid, uintptr_t addr, std::string &result);

bool readSingleList(pid_t pid, uintptr_t address, std::string &result);

bool readDictionary(pid_t pid, uintptr_t address, std::string &result, std::string(*callback)(uint16_t, unsigned char*), size_t objSize = 0x18);

#endif