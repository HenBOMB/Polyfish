#!/usr/bin/env bash
# ==============================================================================
# GPU Architecture & Compute Capability Resolver
# ==============================================================================
# Translates user-friendly GPU model strings (e.g. "5090", "3090 ti", "4090", 
# "A6000", "H100") or raw capability numbers ("120", "86", "8.6") into 
# standard CUDA compute capability integers for Candle and NVCC.
# ==============================================================================

resolve_cuda_cap() {
    local input="${1:-}"
    if [ -z "$input" ]; then
        echo ""
        return 1
    fi

    # Lowercase and strip whitespace, underscores, dashes, and leading "sm"
    local clean
    clean=$(echo "$input" | tr '[:upper:]' '[:lower:]' | tr -d ' _-')
    clean="${clean#sm}"

    case "$clean" in
        *5090*|*5080*|*blackwell*|120|12.0)
            echo "120"
            ;;
        *4090*|*4080*|*4070*|*4000ada*|*pro4000*|*6000ada*|*pro6000*|*ada*|89|8.9)
            echo "89"
            ;;
        *3090*|*3080*|*3070*|*a6000*|*a40*|*a5000*|*ampere*|86|8.6)
            echo "86"
            ;;
        *a100*|80|8.0)
            echo "80"
            ;;
        *h100*|*h200*|*hopper*|90|9.0)
            echo "90"
            ;;
        *2080*|*t4*|*titanrtx*|*turing*|75|7.5)
            echo "75"
            ;;
        *v100*|*volta*|70|7.0)
            echo "70"
            ;;
        [0-9][0-9]|[0-9][0-9][0-9])
            echo "$clean"
            ;;
        *)
            echo ""
            return 1
            ;;
    esac
}

# Generates clean, predictable Docker image tags based on the input GPU or capability
resolve_image_tag() {
    local input="${1:-}"
    local cap="${2:-}"

    local clean
    clean=$(echo "$input" | tr '[:upper:]' '[:lower:]' | tr -d ' _-')
    clean="${clean#sm}"

    case "$clean" in
        *5090*|*5080*|*blackwell*) echo "henbomb/polyfish:5090" ;;
        *4090*|*4080*|*4070*) echo "henbomb/polyfish:4090" ;;
        *3090*|*3080*|*3070*) echo "henbomb/polyfish:3090" ;;
        *a6000*) echo "henbomb/polyfish:a6000" ;;
        *a40*) echo "henbomb/polyfish:a40" ;;
        *h100*|*h200*|*hopper*) echo "henbomb/polyfish:h100" ;;
        *a100*) echo "henbomb/polyfish:a100" ;;
        *)
            case "$cap" in
                120) echo "henbomb/polyfish:5090" ;;
                89) echo "henbomb/polyfish:4090" ;;
                86) echo "henbomb/polyfish:3090" ;;
                90) echo "henbomb/polyfish:h100" ;;
                80) echo "henbomb/polyfish:a100" ;;
                *) echo "henbomb/polyfish:sm${cap}" ;;
            esac
            ;;
    esac
}
