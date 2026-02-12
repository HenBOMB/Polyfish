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

def is_valid_addr(addr, sorted_maps):
    lo = 0
    hi = len(sorted_maps) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        m = sorted_maps[mid]
        if m["start"] <= addr < m["end"]:
            return True
        if addr < m["start"]:
            hi = mid - 1
        else:
            lo = mid + 1
    return False

def read_mem(pid, addr, size):
    try:
        with open(f"/proc/{pid}/mem", "rb", 0) as f:
            f.seek(addr)
            return f.read(size)
    except:
        return None

def scan():
    if len(sys.argv) < 2:
        print("Usage: python3 find_gm.py <pid> [target_turn]")
        return
    
    pid = int(sys.argv[1])
    target_turn = int(sys.argv[2]) if len(sys.argv) > 2 else -1
    all_maps = get_maps(pid)
    sorted_maps = sorted(all_maps, key=lambda x: x["start"])
    
    ga_base = 0
    for m in all_maps:
        if "GameAssembly.dll" in m["pathname"]:
            ga_base = m["start"]
            break
            
    if not ga_base:
        print("GameAssembly.dll not found")
        return

    print(f"GameAssembly.dll base: {hex(ga_base)}")

    # 1. Find GameManager instance
    print("Finding GameManager instances...")
    instances = []
    total_segments = len(all_maps)
    for seg_idx, m in enumerate(all_maps):
        if "rw" not in m["perms"] or (m["end"] - m["start"]) > 200 * 1024 * 1024:
            continue
        
        if seg_idx % 50 == 0:
            print(f"Scanning segment {seg_idx}/{total_segments} ({hex(m['start'])}) - Found {len(instances)} candidates so far...")

        data = read_mem(pid, m["start"], m["end"] - m["start"])
        if not data: continue
        
        for i in range(0, len(data) - 128, 8):
            cb_ptr = int.from_bytes(data[i+0x28 : i+0x30], "little")
            if cb_ptr == 0: continue
            if not is_valid_addr(cb_ptr, sorted_maps): continue
            
            gs_ptr_data = read_mem(pid, cb_ptr + 0x38, 8)
            if not gs_ptr_data: continue
            gs_ptr = int.from_bytes(gs_ptr_data, "little")
            if not is_valid_addr(gs_ptr, sorted_maps): continue
            
            turn_data = read_mem(pid, gs_ptr + 0x18, 4)
            if not turn_data: continue
            turn = int.from_bytes(turn_data, "little")
            
            if target_turn != -1 and turn != target_turn:
                continue

            # NEW: Check player stats if turn matches
            ps_list_ptr_data = read_mem(pid, gs_ptr + 0x38, 8) # GameState -> playerStates
            if not ps_list_ptr_data: continue
            ps_list_ptr = int.from_bytes(ps_list_ptr_data, "little")
            if not is_valid_addr(ps_list_ptr, sorted_maps): continue
            
            count_data = read_mem(pid, ps_list_ptr + 0x18, 4)
            if not count_data: continue
            count = int.from_bytes(count_data, "little")
            if count == 0 or count > 32: continue
            
            has_matching_player = False
            for p_idx in range(count):
                p_ptr_data = read_mem(pid, ps_list_ptr + 0x20 + p_idx * 8, 8)
                if not p_ptr_data: continue
                p_ptr = int.from_bytes(p_ptr_data, "little")
                if not is_valid_addr(p_ptr, sorted_maps): continue
                
                stats_data = read_mem(pid, p_ptr + 0x9C, 8)
                if not stats_data: continue
                currency = int.from_bytes(stats_data[0:4], "little")
                score = int.from_bytes(stats_data[4:8], "little")
                
                if currency == 1 and score == 1030:
                    has_matching_player = True
                    break
            
            if not has_matching_player:
                continue

            ai = int.from_bytes(data[i+0x50 : i+0x54], "little")
            human = int.from_bytes(data[i+0x54 : i+0x58], "little")
            
            inst_addr = m["start"] + i
            print(f"Candidate Match at {hex(inst_addr)} (Turn: {turn}, AI: {ai}, Human: {human}) - PLAYER MATCH FOUND!")
            instances.append(inst_addr)

    if not instances:
        print("No instances found with matching turn and player stats.")
        return

    # 2. Find X such that [X + 0xB8] == G
    print(f"Finding intermediate structs for {len(instances)} candidates...")
    instances_set = set(instances)
    intermediate = {}
    for m in all_maps:
        if "rw" not in m["perms"] or (m["end"] - m["start"]) > 200 * 1024 * 1024:
            continue
        data = read_mem(pid, m["start"], m["end"] - m["start"])
        if not data: continue
        for i in range(0, len(data) - 0xC0, 8):
            val_at_b8 = int.from_bytes(data[i + 0xB8 : i + 0xC0], "little")
            if val_at_b8 in instances_set:
                x_addr = m["start"] + i
                print(f"Found struct X at {hex(x_addr)} pointing to {hex(val_at_b8)}")
                intermediate[x_addr] = val_at_b8

    if not intermediate:
        print("No intermediate structs found.")
        return

    # 3. Find static pointers in GameAssembly.dll pointing to X
    print("Finding static offset in GameAssembly.dll...")
    intermediate_set = set(intermediate.keys())
    # Note: increased search range for GA segments
    ga_segments = [m for m in all_maps if "GameAssembly" in m["pathname"] or (m["start"] >= ga_base and m["start"] < ga_base + 0x40000000 and (m["pathname"] == "" or m["pathname"].endswith(".dll")))]
    
    for m in ga_segments:
        if "r" not in m["perms"]: continue
        print(f"Scanning segment {hex(m['start'])} ({m['perms']})...")
        data = read_mem(pid, m["start"], m["end"] - m["start"])
        if not data: continue
        for i in range(0, len(data) - 8, 8):
            val = int.from_bytes(data[i : i+8], "little")
            if val in intermediate_set:
                addr = m["start"] + i
                offset = addr - ga_base
                print(f"\n[!!!] SUCCESS! found static offset: GameAssembly.dll + {hex(offset)}")
                print(f"Path: modBase + {hex(offset)} -> {hex(val)} -> [{hex(val)} + 0xB8] -> {hex(intermediate[val])}")

if __name__ == "__main__":
    scan()
