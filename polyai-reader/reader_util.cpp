#include "reader_util.h"
#include "reader_types.h"
#include <sys/uio.h>
#include <random>
#include <unistd.h>
#include <chrono>
#include <thread>

static std::random_device rd;
static std::mt19937 gen(rd());

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

std::string readString(pid_t pid, uintptr_t addr) {
    uint16_t wstrcount = 0;

    uintptr_t lengthAddr = getPlace(pid, addr, { 0x10 });
    if (!readPiece(pid, lengthAddr, wstrcount) || wstrcount == 0) {
        return "";
    }

    uintptr_t strDataAddr = getPlace(pid, addr, { 0x14 });
    if (strDataAddr == 0) {
        return "";
    }

    std::wstring wstr;
    wstr.reserve(wstrcount);

    for (size_t i = 0; i < wstrcount; i++) {
        uint16_t wchar;
        if (!readPiece(pid, strDataAddr + (i * 2), wchar)) {
            break;
        }
        wchar = le16toh(wchar);
        if (wchar == 0) break;
        wstr += static_cast<wchar_t>(wchar);
    }

    std::string out;

    for (wchar_t wc : wstr) {
        if (wc == 0) break;
        if (wc < 0x80) {
            out += static_cast<char>(wc);
        } else if (wc < 0x800) {
            out += static_cast<char>(0xC0 | (wc >> 6));
            out += static_cast<char>(0x80 | (wc & 0x3F));
        } else {
            out += static_cast<char>(0xE0 | (wc >> 12));
            out += static_cast<char>(0x80 | ((wc >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (wc & 0x3F));
        }
    }

    return out;
}

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

bool readSplitValue(pid_t pid, uintptr_t address, uint16_t &value1, uint16_t &value2) {
    uint64_t coordinates = 0;
    if (!readPiece(pid, address, coordinates)) {
        return false;
    }
    value1 = static_cast<uint16_t>(coordinates);
    value2 = static_cast<uint16_t>(coordinates >> 32);
    return true;
}

bool readSingleList(pid_t pid, uintptr_t address, std::string &result) {
    uint16_t size;
    if (!readPiece(pid, getPlace(pid, address, { 0x18 }), size)) {
        return false;
    }

    size_t maxSize = std::min(static_cast<size_t>(size), size_t(100));
    if (maxSize == 0) {
        return false;
    }

    size_t bufferSize = maxSize * 0x4;
    std::vector<unsigned char> listBuffer(bufferSize);

    uintptr_t listBase = getPlace(pid, address, { 0x10, 0x20 });
    if (!listBase || !readBlock(pid, listBase, listBuffer.data(), bufferSize)) {
        return false;
    }

    for (uint32_t i = 0; i < maxSize; i++) {
        uint16_t value = *(uint16_t*)&listBuffer[i * 0x4];
        result += std::to_string(value) + "&";
    }
    if (!result.empty()) result.pop_back();

    return true;
}