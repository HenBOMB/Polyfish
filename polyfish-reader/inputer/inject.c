// inject.c
#include <stdint.h>

// Define a constructor function that runs on load
void __attribute__((constructor)) init(void) {
    void *actionManager = (void *)0x693F3880;  // ActionManager pointer
    void *error = "";                        // Null error pointer

    // Allocate space on the stack for ResearchCommand (0x20 bytes, rounded)
    char buffer[32] __attribute__((aligned(16)));  // 32 bytes, 16-byte aligned
    void *command = buffer;                        // Pointer to our object

    // Assembly to build ResearchCommand and call ExecuteCommand
    asm volatile (
        "movq $0x6404fff0, %%rax\n\t"    // ResearchCommand vtable (64-bit)
        "movq %%rax, (%%rsi)\n\t"        // Set vtable at command[0]
        "movb $1, 0x10(%%rsi)\n\t"       // playerId = 1 at command[0x10]
        "movl $19, 0x18(%%rsi)\n\t"      // techId = 42 at command[0x18]
        "movq %0, %%rdi\n\t"             // ActionManager pointer
        "movq %1, %%rdx\n\t"             // Error pointer (null)
        "movq $0x63b95308, %%rax\n\t"    // ExecuteCommand address
        "call *%%rax\n\t"                // Call it
        :                                // No outputs
        : "r"(actionManager), "r"(error), "S"(command)  // Inputs: RSI = command
        : "rax", "rdi", "rdx", "memory"  // Clobbered registers
    );
}

// Optional: Prevent stripping
void dummy(void) {
    init();
}