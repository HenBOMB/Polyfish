#include "reader_util.h"
#include "reader_types.h"
#include <sys/uio.h>
#include <random>
#include <unistd.h>
#include <chrono>
#include <thread>
#include <iostream>

static std::random_device rd;
static std::mt19937 gen(rd());

uintptr_t getPlace(pid_t pid, uintptr_t base, const std::vector<uintptr_t>& offsets) {
    uintptr_t addr = base;
    for (size_t i = 0; i < offsets.size(); ++i) {
        if (!readPiece(pid, addr, addr)) {
            return 0;
        }
        addr += offsets[i];
    }
    return addr;
}

bool readDictionary(pid_t pid, uintptr_t address,  std::string &result, std::string(*callback)(uint16_t, unsigned char*), size_t objSize) {
    uint16_t size;

    if (!readPiece(pid, getPlace(pid, address, { _0x_IN_DICT_COUNT }), size)) {
        return false;
    }

    if (size == 0) {
        result = "";
        return true;
    }

    size_t dictItemSize = 0x18;
    size_t bufferSize = size * dictItemSize;
    unsigned char dictBuffer[bufferSize];

    uintptr_t dictBase = getPlace(pid, address, { _0x_IN_DICT_ENTRIES });
    if (!readBlock(pid, dictBase, dictBuffer, bufferSize)) {
        return false;
    }
    
    // std::cout << "[address]: 0x" << std::hex << dictBase << std::endl << std::dec;

    for (uint32_t i = 0; i < size; i++) {
        uint8_t key;
        unsigned char valueBuff[objSize + _0x_TRAILING_OFFSET];
        
        if (!readPiece(pid, getPlace(pid, dictBase, { _0x_IN_DICT_KEY_SHIFT + i * dictItemSize }), key) ||
            !readBlock(pid, getPlace(pid, dictBase, { _0x_IN_DICT_VAL_SHIFT + i * dictItemSize, 0x0 }), valueBuff, objSize)) {
            break;
        }

        std::string data = callback(key, valueBuff);
        result += data + "&";
    }

    result.pop_back();

    return true;
}

bool readBlock(pid_t pid, uintptr_t addr, void* buffer, size_t size) {
    static std::uniform_int_distribution<> dis(69, 333);
    std::this_thread::sleep_for(std::chrono::microseconds(dis(gen)));

    struct iovec local[1];
    struct iovec remote[1];

    local[0].iov_base = buffer;
    local[0].iov_len = size;
    remote[0].iov_base = (void*)addr;
    remote[0].iov_len = size;

    ssize_t nread = process_vm_readv(pid, local, 1, remote, 1, 0);
    return nread == size;
}

bool readWord(pid_t pid, uintptr_t addr, std::string &result) {
    unsigned char buff[0x14 + 0x10];

    if (!readPiece(pid, addr, buff)) {
        return false;
    }

    int length = *(int16_t*)&buff[0x10];

    result = std::string((char*)&buff[0x14], 0x2 * length);

    return true;
}

bool readSingleList(pid_t pid, uintptr_t address, std::string &result) {
    uint16_t size;

    if (!readPiece(pid, getPlace(pid, address, { 0x18 }), size)) {
        return false;
    }

    if (size == 0) {
        result = "";
        return true;
    }

    size_t bufferSize = size * 0x4;
    std::vector<unsigned char> listBuffer(bufferSize);

    uintptr_t listBase = getPlace(pid, address, { 0x10, 0x20 });
    if (!listBase || !readBlock(pid, listBase, listBuffer.data(), bufferSize)) {
        return false;
    }

    for (uint32_t i = 0; i < size; i++) {
        uint16_t value = *(uint16_t*)&listBuffer[i * 0x4];
        result += std::to_string(value) + "&";
    }

    // Remove the last ampersand
    if (!result.empty()) result.pop_back();

    return true;
}