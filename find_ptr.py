import sys
import os
import re

def get_maps(pid):
    maps = []
    with open(f"/proc/{pid}/maps", "r") as f:
        for line in f:
            m = re.match(r"([0-9a-f]+)-([0-9a-f]+)\s+([rwxp-]+)\s+([0-9a-f]+)\s+([0-9a-f:]+)\s+(\d+)\s*(.*)", line)
            if m:
                start = int(m.group(1), 16)
                end = int(m.group(2), 16)
                perms = m.group(3)
                pathname = m.group(7).strip()
                maps.append({"start": start, "end": end, "perms": perms, "pathname": pathname})
    return maps

def read_mem(pid, addr, size):
    try:
        with open(f"/proc/{pid}/mem", "rb", 0) as f:
            f.seek(addr)
            return f.read(size)
    except:
        return None

def scan():
    if len(sys.argv) < 3:
        print("Usage: python3 find_ptr.py <pid> <target_addr>")
        return
    
    pid = int(sys.argv[1])
    target = int(sys.argv[2], 16)
    maps = get_maps(pid)
    
    ga_base = 0
    for m in maps:
        if "GameAssembly.dll" in m["pathname"]:
            ga_base = m["start"]
            break
            
    # Search for pointers to target
    # We are looking for P such that [P + 0xB8] == target
    # OR [P] + 0xB8 == target (so [P] == target - 0xB8)
    
    print(f"Scanning for pointers that lead to {hex(target)}...")
    
    # Let's search for pointers to (target - 0xB8) or similar
    # or just any pointers to target
    
    for m in maps:
        if "r" not in m["perms"]: continue
        
        data = read_mem(pid, m["start"], m["end"] - m["start"])
        if not data: continue
        
        for i in range(0, len(data) - 8, 8):
            val = int.from_bytes(data[i : i+8], "little")
            
            # Case 1: val points to target directly
            if val == target:
                addr = m["start"] + i
                rem = ""
                if ga_base and addr >= ga_base and addr < ga_base + 0x10000000:
                    rem = f" (GA offset: {hex(addr - ga_base)})"
                print(f"Found pointer to {hex(target)} at {hex(addr)} {m['pathname']}{rem}")
            
            # Case 2: [val + 0xB8] == target
            # This means val is the address of some struct whose field at 0xB8 is target
            # So we check if [val + 0xB8] is readable and equal to target
            
            # This is slow if we do it for every val. Let's only do it if val looks like a valid pointer.
            # actually we can just find where (target - 0xB8) is potentially stored? 
            # No, we want val such that [val + 0xB8] == target.
            # So search for val such that [val + 0xB8] == target.
            
    # Let's also search for Turn: 1, AI: 1, Human: 1 instances again but more precisely
    
if __name__ == "__main__":
    scan()
