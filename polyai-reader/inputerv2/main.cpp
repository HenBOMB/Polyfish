#include <pthread.h>
#include <dlfcn.h>
#include <unistd.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

static const uintptr_t FN_ADDR = 0x318ECD00ULL;   // <- REPLACE with your native JIT address
static const uintptr_t INSTANCE_ADDR = 0x711F1900ULL; // <- REPLACE with ClientInteraction instance
static const uintptr_t TILE_ADDR = 0x6E0888C0ULL;     // <- REPLACE with Tile object

// Simple logging helper
static void log_msg(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    int fd = open("/tmp/call_selecttile.log", O_WRONLY | O_CREAT | O_APPEND, 0644);
    if (fd >= 0) {
        char buf[1024];
        int n = vsnprintf(buf, sizeof(buf), fmt, ap);
        if (n > 0) {
            write(fd, buf, (size_t)n);
            write(fd, "\n", 1);
        }
        close(fd);
    }
    va_end(ap);
}

// Thread routine that performs the call
static void* thread_main(void* arg) {
    (void)arg;
    // small wait so the game initializes
    sleep(1);

    log_msg("call_selecttile.so: thread started. FN_ADDR=0x%lx INSTANCE=0x%lx TILE=0x%lx",
            (unsigned long)FN_ADDR, (unsigned long)INSTANCE_ADDR, (unsigned long)TILE_ADDR);

    // Validate addresses a bit (not guaranteed, but helpful)
    if (FN_ADDR == 0 || INSTANCE_ADDR == 0 || TILE_ADDR == 0) {
        log_msg("call_selecttile.so: one or more addresses are zero; aborting call.");
        return nullptr;
    }

    // Create a function pointer with SysV ABI: first arg in RDI, second in RSI
    using FnType = void(*)(void*, void*);
    FnType fn = reinterpret_cast<FnType>(FN_ADDR);

    // Attempt the call
    // Surround with safeguards: use volatile to avoid surprising optimizations
    volatile void* inst = reinterpret_cast<void*>(INSTANCE_ADDR);
    volatile void* tile = reinterpret_cast<void*>(TILE_ADDR);

    // Try/catch won't catch native crashes; we at least log before/after
    log_msg("call_selecttile.so: invoking function...");
    // call
    fn((void*)inst, (void*)tile);
    log_msg("call_selecttile.so: function returned (if it returned).");

    // keep library loaded for a bit (optional)
    // sleep(1);

    return nullptr;
}

// constructor attribute: runs when the SO is loaded (e.g. LD_PRELOAD or dlopen)
__attribute__((constructor))
static void on_load(void) {
    log_msg("call_selecttile.so: on_load entry.");

    pthread_t thr;
    pthread_attr_t attr;
    pthread_attr_init(&attr);

    // create detached thread so it does not need to be joined
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);

    int r = pthread_create(&thr, &attr, thread_main, nullptr);
    if (r != 0) {
        log_msg("call_selecttile.so: pthread_create failed: %d (%s)", r, strerror(r));
    } else {
        log_msg("call_selecttile.so: worker thread created.");
    }

    pthread_attr_destroy(&attr);
}

__attribute__((destructor))
static void on_unload(void) {
    log_msg("call_selecttile.so: on_unload.");
}
