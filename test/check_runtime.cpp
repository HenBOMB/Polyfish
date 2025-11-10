// get_gamemgr_instance.cpp
// Build: g++ -fPIC -shared -o get_instance.so get_gamemgr_instance.cpp -ldl
// Usage: inject / LD_PRELOAD into the target game process and call run_resolve()
// Example: inside your injected code call run_resolve(0x318BC1B0);

#include <dlfcn.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

using MonoClass        = void;
using MonoClassField   = void;
using MonoDomain       = void;
using MonoObject       = void;

using t_mono_class_get_field_from_name = MonoClassField*(*)(MonoClass*, const char*);
using t_mono_class_get_static_field_data = void*(*)(MonoClass*);
using t_mono_field_get_value = void(*)(void* /*obj_or_static_data*/, MonoClassField*, void* /*out*/);
using t_mono_class_get_fields = MonoClassField*(*)(MonoClass*, void**);
using t_mono_field_get_name = const char*(*)(MonoClassField*);

struct MonoApi {
    void* handle = nullptr;
    t_mono_class_get_field_from_name mono_class_get_field_from_name = nullptr;
    t_mono_class_get_static_field_data mono_class_get_static_field_data = nullptr;
    t_mono_field_get_value mono_field_get_value = nullptr;
    t_mono_class_get_fields mono_class_get_fields = nullptr;
    t_mono_field_get_name mono_field_get_name = nullptr;
};

static MonoApi mono;

bool load_mono_library() {
    const char* names[] = {
        "libmono.so",
        "libmono-2.0.so",
        "libmono-bdwgc.so",
        nullptr
    };

    for (const char** p = names; *p; ++p) {
        void* h = dlopen(*p, RTLD_NOW | RTLD_NOLOAD);
        if (!h) {
            // try to open without NOLOAD (in case it's not already loaded)
            h = dlopen(*p, RTLD_NOW);
        }
        if (h) {
            mono.handle = h;
            break;
        }
    }
    if (!mono.handle) {
        fprintf(stderr, "[mono] couldn't dlopen any known libmono names\n");
        return false;
    }

    mono.mono_class_get_field_from_name = (t_mono_class_get_field_from_name)dlsym(mono.handle, "mono_class_get_field_from_name");
    mono.mono_class_get_static_field_data = (t_mono_class_get_static_field_data)dlsym(mono.handle, "mono_class_get_static_field_data");
    mono.mono_field_get_value = (t_mono_field_get_value)dlsym(mono.handle, "mono_field_get_value");
    mono.mono_class_get_fields = (t_mono_class_get_fields)dlsym(mono.handle, "mono_class_get_fields");
    mono.mono_field_get_name = (t_mono_field_get_name)dlsym(mono.handle, "mono_field_get_name");

    // mono_class_get_field_from_name and mono_class_get_static_field_data and mono_field_get_value are the most important
    if (!mono.mono_class_get_static_field_data || !mono.mono_field_get_value) {
        fprintf(stderr, "[mono] required symbols not found (mono_class_get_static_field_data or mono_field_get_value)\n");
        return false;
    }

    // mono_class_get_field_from_name is optional (we can fallback to iterate fields)
    return true;
}

uintptr_t resolve_gamemanager_instance_from_class(uintptr_t mono_class_addr) {
    if (!mono.handle) {
        if (!load_mono_library()) return 0;
    }

    MonoClass* klass = (MonoClass*)(uintptr_t)mono_class_addr;
    MonoClassField* field = nullptr;

    if (mono.mono_class_get_field_from_name) {
        field = mono.mono_class_get_field_from_name(klass, "instance");
    }

    // fallback: iterate fields and match name == "instance"
    if (!field && mono.mono_class_get_fields && mono.mono_field_get_name) {
        void* iter = nullptr;
        MonoClassField* f = nullptr;
        while ((f = mono.mono_class_get_fields(klass, &iter)) != nullptr) {
            const char* fname = mono.mono_field_get_name(f);
            if (fname && strcmp(fname, "instance") == 0) {
                field = f;
                break;
            }
        }
    }

    if (!field) {
        fprintf(stderr, "[mono] couldn't find field 'instance' in class at %p\n", klass);
        return 0;
    }

    void* static_data = nullptr;
    if (mono.mono_class_get_static_field_data) {
        static_data = mono.mono_class_get_static_field_data(klass);
    }
    if (!static_data) {
        fprintf(stderr, "[mono] mono_class_get_static_field_data returned NULL\n");
        return 0;
    }

    uintptr_t instance_addr = 0;
    mono.mono_field_get_value(static_data, field, &instance_addr);

    return instance_addr;
}

// Convenience entry point for calling from an injected environment
extern "C" void run_resolve_and_print(uintptr_t mono_class_addr) {
    uintptr_t inst = resolve_gamemanager_instance_from_class(mono_class_addr);
    if (inst) {
        printf("[+] Resolved GameManager.instance = 0x%zx\n", (size_t)inst);
    } else {
        printf("[-] Failed to resolve GameManager.instance\n");
    }
}

// If you want a main for testing when running inside a process (not recommended as a standalone):
#ifdef TEST_MAIN
int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <GameManager_MonoClass_address_hex>\n", argv[0]);
        return 1;
    }
    uintptr_t klass = strtoull(argv[1], nullptr, 16);
    run_resolve_and_print(klass);
    return 0;
}
#endif
