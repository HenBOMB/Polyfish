#ifndef READER_UTIL_H
#define READER_UTIL_H

#include "reader_types.h"

#include <vector>
#include <sys/uio.h>
#include <cstdint>

bool readBlock(pid_t pid, uintptr_t addr, void* buffer, size_t size);

template <typename T>
bool readPiece(pid_t pid, uintptr_t addr, T &value) {
    return readBlock(pid, addr, &value, sizeof(T));
}

uintptr_t getPlace(pid_t pid, uintptr_t base, const std::vector<uintptr_t>& offsets);

std::string readString(pid_t pid, uintptr_t addr);

bool readSplitValue(pid_t pid, uintptr_t address, uint16_t &value1, uint16_t &value2);

bool readSingleList(pid_t pid, uintptr_t address, std::string &result);

#endif