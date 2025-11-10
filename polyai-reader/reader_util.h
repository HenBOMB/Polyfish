#ifndef READER_UTIL_H
#define READER_UTIL_H

#include "reader_types.h"
#include <vector>
#include <sys/uio.h>
#include <cstdint>
#include <string>
#include <cstdint>


uintptr_t getPlace(pid_t pid, uintptr_t base, const std::vector<uintptr_t>& offsets);

bool readBlock(pid_t pid, uintptr_t addr, void* buffer, size_t size);

template <typename T>
bool readPiece(pid_t pid, uintptr_t addr, T &result) {
    return readBlock(pid, addr, &result, sizeof(T));
}

// template<typename A, typename B>
// void readBuff(const unsigned char* b, A e, B &result) {
//     result = *(E*)&b[static_cast<std::size_t>(e)];
// }

bool readWord(pid_t pid, uintptr_t addr, std::string &result);

bool readSingleList(pid_t pid, uintptr_t address, std::string &result);

bool readDictionary(pid_t pid, uintptr_t address, std::string &result, std::string(*callback)(uint16_t, unsigned char*), std::size_t objSize);

template <typename E>
bool readSingleListMagic(pid_t pid, uintptr_t address, std::vector<E> &result) {
    uint16_t size;

    if (!readPiece(pid, getPlace(pid, address, { 0x18 }), size)) {
        return false;
    }

    if (size == 0 || size > 1000) {
        return true;
    }

    uintptr_t listBase = getPlace(pid, address, { 0x10 });

    for (uint32_t i = 0; i < size; i++) {
        E value;
        readPiece(pid, getPlace(pid, listBase, { 0x20 + i * sizeof(E) }), value);
        result.insert(result.end(), value);
    }

    return true;
}

template <typename ValueT, typename Callback>
bool readDictionaryMagic(pid_t pid, uintptr_t address, std::unordered_map<uint16_t, ValueT> &result,Callback callback, std::size_t objSize) {
    uint16_t size;

    if (!readPiece(pid, getPlace(pid, address, { 0x20 }), size)) {
        return false;
    }

    if (size == 0) {
        result = { };
        return true;
    }

    // std::cout << "size: " << size << std::endl;

    size_t dictItemSize = 0x18;
    size_t bufferSize = size * dictItemSize;
    unsigned char dictBuffer[bufferSize];

    uintptr_t dictBase = getPlace(pid, address, { 0x18 });
    if (!readBlock(pid, dictBase, dictBuffer, bufferSize)) {
        return false;
    }
    
    // std::cout << "[address]: 0x" << std::hex << dictBase << std::endl << std::dec;

    for (uint32_t i = 0; i < size; i++) {
        uint16_t key;
        unsigned char valueBuff[objSize + _0x_TRAILING_OFFSET];
        
        if (!readPiece(pid, getPlace(pid, dictBase, { 0x20 + i * dictItemSize }), key) ||
            !readBlock(pid, getPlace(pid, dictBase, { 0x30 + i * dictItemSize, 0x0 }), valueBuff, objSize)) {
            break;
        }

        if (key < 0 || key >= 255) {
            break;
        }

        // std::cout << "key: " << key << std::endl;

        result[key] = callback(key, valueBuff);
    }

    return true;
}



#endif
