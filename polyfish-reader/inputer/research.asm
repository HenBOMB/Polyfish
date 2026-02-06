; research.asm
section .text
global main
extern printf

main:
    ; Allocate stack space
    sub rsp, 0x30           ; 0x20 for ResearchCommand, 0x8 for error, 0x8 align

    ; Build ResearchCommand at RSP
    mov rax, 0x6404fff0     ; ResearchCommand vtable
    mov [rsp], rax
    mov byte [rsp+0x10], 1  ; playerId
    mov dword [rsp+0x18], 19; techId

    ; Error string pointer
    mov qword [rsp+0x20], 0 ; Null initially

    ; Arguments
    mov rdi, 0x693F3880     ; ActionManager
    mov rsi, rsp            ; ResearchCommand pointer
    lea rdx, [rsp+0x20]     ; &errorString

    ; Call ExecuteCommand
    mov rax, 0x63b95308
    call rax

    ; Check result
    test rax, rax
    jz .failure

    ; Success
    mov rdi, success_msg
    call printf
    jmp .exit

.failure:
    mov rax, [rsp+0x20]
    test rax, rax
    jz .no_error
    mov rdi, failure_msg
    mov rsi, [rax]          ; Error string
    call printf
    jmp .exit

.no_error:
    mov rdi, no_error_msg
    call printf

.exit:
    add rsp, 0x30
    xor eax, eax
    ret

section .data
success_msg: db "Success!", 10, 0
failure_msg: db "Failed: %s", 10, 0
no_error_msg: db "Failed with no error message.", 10, 0