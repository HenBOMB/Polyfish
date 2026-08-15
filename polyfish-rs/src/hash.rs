//! xxHash32 implementation for deterministic Polytopia RNG.

const PRIME32_1: u32 = 2654435761;
const PRIME32_2: u32 = 2246822519;
const PRIME32_3: u32 = 3266489917;
const PRIME32_4: u32 = 668265263;
const PRIME32_5: u32 = 374761393;

/// Computes the 32-bit xxHash of the input data using the given seed.
pub fn xxhash32(input: &[u8], seed: u32) -> u32 {
    let mut h32: u32;
    let len = input.len();

    if len >= 16 {
        let limit = len - 16;
        let mut v1 = seed.wrapping_add(PRIME32_1).wrapping_add(PRIME32_2);
        let mut v2 = seed.wrapping_add(PRIME32_2);
        let mut v3 = seed.wrapping_add(0);
        let mut v4 = seed.wrapping_sub(PRIME32_1);

        let mut offset = 0;
        while offset <= limit {
            v1 = round(
                v1,
                u32::from_le_bytes(input[offset..offset + 4].try_into().unwrap()),
            );
            v2 = round(
                v2,
                u32::from_le_bytes(input[offset + 4..offset + 8].try_into().unwrap()),
            );
            v3 = round(
                v3,
                u32::from_le_bytes(input[offset + 8..offset + 12].try_into().unwrap()),
            );
            v4 = round(
                v4,
                u32::from_le_bytes(input[offset + 12..offset + 16].try_into().unwrap()),
            );
            offset += 16;
        }

        h32 = v1
            .rotate_left(1)
            .wrapping_add(v2.rotate_left(7))
            .wrapping_add(v3.rotate_left(12))
            .wrapping_add(v4.rotate_left(18));
    } else {
        h32 = seed.wrapping_add(PRIME32_5);
    }

    h32 = h32.wrapping_add(len as u32);

    let mut offset = if len >= 16 { len - (len % 16) } else { 0 };
    while offset + 4 <= len {
        let val = u32::from_le_bytes(input[offset..offset + 4].try_into().unwrap());
        h32 = h32.wrapping_add(val.wrapping_mul(PRIME32_3));
        h32 = h32.rotate_left(17).wrapping_mul(PRIME32_4);
        offset += 4;
    }

    while offset < len {
        let val = input[offset] as u32;
        h32 = h32.wrapping_add(val.wrapping_mul(PRIME32_5));
        h32 = h32.rotate_left(11).wrapping_mul(PRIME32_1);
        offset += 1;
    }

    h32 ^= h32 >> 15;
    h32 = h32.wrapping_mul(PRIME32_2);
    h32 ^= h32 >> 13;
    h32 = h32.wrapping_mul(PRIME32_3);
    h32 ^= h32 >> 16;

    h32
}

#[inline]
fn round(v: u32, input: u32) -> u32 {
    let mut v = v.wrapping_add(input.wrapping_mul(PRIME32_2));
    v = v.rotate_left(13);
    v.wrapping_mul(PRIME32_1)
}

/// Helper that mirrors the game's `GetHash(int[] data)` method for an array of integers.
pub fn get_hash(seed: u32, data: &[i32]) -> u32 {
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for &val in data {
        bytes.extend_from_slice(&val.to_le_bytes());
    }
    xxhash32(&bytes, seed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xxhash32_standard_vectors() {
        assert_eq!(xxhash32(b"", 0), 46947589);
        assert_eq!(xxhash32(b"a", 0), 1426945110);
        assert_eq!(xxhash32(b"Hello World", 0), 2986153710);
        assert_eq!(xxhash32(b"Hello World", 12345), 782531257);
        assert_eq!(
            xxhash32(b"Once upon a time in a galaxy far far away...", 0),
            531163085
        );
    }
}
