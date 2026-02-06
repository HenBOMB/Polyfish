#ifndef READER_UTIL_H
#define READER_UTIL_H

#include "reader_types.h"
#include <vector>
#include <sys/uio.h>
#include <cstdint>
#include <string>
#include <cstdint>
#include <unordered_map>


uintptr_t getPlace(pid_t pid, uintptr_t base, const std::vector<uintptr_t>& offsets);

bool readBlock(pid_t pid, uintptr_t addr, void* buffer, size_t size);

template <typename T>
bool readPiece(pid_t pid, uintptr_t addr, T &result) {
    return readBlock(pid, addr, &result, sizeof(T));
}

inline bool readPiece(pid_t pid, uintptr_t addr, void *buffer, size_t size) {
    return readBlock(pid, addr, buffer, size);
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

static std::string utf16le_to_utf8(const char16_t* src, size_t len) {
    std::string out;
    out.reserve(len * 3 + 1);

    for (size_t i = 0; i < len; ++i) {
        uint32_t cp = static_cast<uint16_t>(src[i]);

        // surrogate pair?
        if (cp >= 0xD800 && cp <= 0xDBFF) { // high surrogate
            if (i + 1 < len) {
                uint32_t lo = static_cast<uint16_t>(src[i+1]);
                if (lo >= 0xDC00 && lo <= 0xDFFF) {
                    cp = 0x10000 + (((cp - 0xD800) << 10) | (lo - 0xDC00));
                    ++i; // consumed low surrogate
                } else {
                    // invalid pair -> replacement char
                    cp = 0xFFFD;
                }
            } else {
                cp = 0xFFFD;
            }
        } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
            // stray low surrogate
            cp = 0xFFFD;
        }

        // encode cp into UTF-8
        if (cp <= 0x7F) {
            out.push_back(static_cast<char>(cp));
        } else if (cp <= 0x7FF) {
            out.push_back(static_cast<char>(0xC0 | ((cp >> 6) & 0x1F)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else if (cp <= 0xFFFF) {
            out.push_back(static_cast<char>(0xE0 | ((cp >> 12) & 0x0F)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else {
            out.push_back(static_cast<char>(0xF0 | ((cp >> 18) & 0x07)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        }
    }

    return out;
}

static bool BRUHMOTHERFUCKINGSHIT(pid_t pid, uintptr_t addr, std::string &result) {
    // tune MAX_CHARS to expected maximum username length in characters
    const size_t MAX_CHARS = 256;
    const size_t RAW_BUF_SIZE = 0x14 + 2 * MAX_CHARS;
    std::vector<unsigned char> buff(RAW_BUF_SIZE);

    // make sure your readPiece reads RAW_BUF_SIZE bytes or more
    if (!readPiece(pid, addr, buff.data(), buff.size())) {
        return false;
    }

    // bounds check for length field at offset 0x10
    if (0x10 + sizeof(uint16_t) > buff.size()) return false;
    uint16_t length = *reinterpret_cast<uint16_t*>(&buff[0x10]);

    if (length == 0) { result.clear(); return true; }
    if ((size_t)length > MAX_CHARS) length = static_cast<uint16_t>(MAX_CHARS);

    // ensure the data for characters exists in buffer
    size_t neededBytes = 0x14 + 2 * length;
    if (neededBytes > buff.size()) return false;

    // Build char16_t array interpreting bytes as UTF-16LE
    std::vector<char16_t> u16;
    u16.reserve(length);
    const unsigned char* dataStart = &buff[0x14];

    for (size_t i = 0; i < length; ++i) {
        size_t off = i * 2;
        uint16_t lo = dataStart[off];
        uint16_t hi = dataStart[off + 1];
        uint16_t val = static_cast<uint16_t>(lo | (hi << 8)); // little-endian
        u16.push_back(static_cast<char16_t>(val));
    }

    // strip BOM if present
    if (!u16.empty() && u16[0] == 0xFEFF) {
        u16.erase(u16.begin());
    }

    // convert to UTF-8
    result = utf16le_to_utf8(u16.data(), u16.size());
    return true;
}

#endif
