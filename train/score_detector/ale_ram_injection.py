# Copyright 2025 Keen Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

"""
Game Ram Config
--------

To add a new game:

Each ale game in src/games/supported defines a 'step' function
where score and lives handling is found. For score, look for:

int score = getDecimalScore(ADDR1, ADDR2, ..., &system);
"score_addr": [ADDR1, ADDR2, ...],
"score_type": "packed_bcd",  # most common
** Byte order depends on param order in getDecimalScore(...), often MSB -> LSB, but the function reads **RAM in that order**, so this becomes the "score_addr" order.

"bcd_order": How **RAM addresses** are ordered (lsb or msb)
"digit_order": How **digits** are packed inside each byte (lsb or msb)

If ALE has:
getDecimalScore(ADDR_HI, ADDR_MID, ADDR_LO, &system);

Then:
"score_addr": [ADDR_HI, ADDR_MID, ADDR_LO],
"bcd_order": "lsb",       # RAM goes low-to-high
"digit_order": "msb"      # digits packed left-to-right (normal)

If rendered score is off by x10 or x100, "digit_order" is probably wrong. If digits are reversed, "bcd_order" is wrong.

If ALE has:
int lives = readRam(&system, ADDR);

"lives_addr": ADDR,
"lives_nibble": "high" or "low",  # based on shift
"lives_offset": 1                # if +1 is applied

if its raw:
"lives_nibble": "low",
"lives_offset": 0
"""

GAME_RAM_CONFIG = {
    "atlantis": {
        "score_addr": [0xA1, 0xA3, 0xA2],
        "lives_addr": 0xF1,
        "lives_nibble": "low",
        "score_type": "packed_bcd",
        "bcd_order": "msb",
        "digit_order": "msb",
        "score_multiplier": 100,
        "total_lives": 6,
        "display_lives": 6,
        "score_step": [100],
        "max_score": 84300,  # capped at 84400 with ram injection (84500 but isn't correct); is this the max?
        "score_digits": 6,
    },
    "battle_zone": {
        "score_addr": [0x9E, 0x9D],
        "lives_addr": 0xBA,
        "lives_nibble": "low",
        "score_type": "custom_battlezone",
        "score_multiplier": 1000,
        "score_step": [1000],
        "max_score": 501000,  # capped
        "total_lives": 5,
        "display_lives": 5,
        "score_digits": 6,
    },
    "centipede": {
        "score_addr": [0xF6, 0xF5, 0xF4],
        "lives_addr": 0xED,
        "score_type": "packed_bcd",
        "bcd_order": "lsb",
        "digit_order": "msb",
        "score_multiplier": 1,
        "lives_nibble": "high",
        "total_lives": 3,
        "display_lives": 2,
        "score_step": [
            1,
        ],
        "max_score": 9999,
        "score_digits": 6,
    },
    "defender": {
        "score_addr": [0x9C, 0x9D, 0x9E, 0x9F, 0xA0, 0xA1],
        "lives_addr": 0xC2,
        "score_type": "custom_defender",
        "lives_nibble": "low",
        "total_lives": 3,
        "display_lives": 3,
        "score_step": [50],
        "max_score": 73000,  # capped at 73000 with ram injection; is this the max?
        "score_multiplier": 1,
        "score_digits": 6,
    },
    "krull": {
        "score_addr": [0x9E, 0x9D, 0x9C],
        "lives_addr": 0x9F,
        "lives_nibble": "low",
        "score_type": "packed_bcd",
        "bcd_order": "lsb",
        "digit_order": "msb",
        "total_lives": 3,
        "display_lives": 2,
        "score_step": [10],
        "max_score": 99990,
        "score_digits": 6,
    },
    "ms_pacman": {
        "score_addr": [0xF8, 0xF9, 0xFA],
        "lives_addr": 0xFB,
        "score_type": "packed_bcd",
        "bcd_order": "lsb",
        "lives_nibble": "low",
        "total_lives": 3,
        "display_lives": 2,
        "score_step": [
            10,
        ],
        "max_score": 99990,
        "score_digits": 6,
    },
    "qbert": {
        "score_addr": [0xD9, 0xDA, 0xDB],
        "lives_addr": 0x88,
        "score_type": "packed_bcd",
        "bcd_order": "msb",
        "lives_signed": True,
        "total_lives": 4,
        "display_lives": 3,
        "score_step": [25],
        "max_score": 99950,
        "score_digits": 5,
    },
    "up_n_down": {
        "score_addr": [0x82, 0x81, 0x80],
        "lives_addr": 0x86,
        "score_type": "packed_bcd",
        "bcd_order": "lsb",
        "digit_order": "msb",
        "lives_offset": 1,
        "total_lives": 5,
        "display_lives": 4,
        "score_step": [10],
        "max_score": 99990,
        "score_digits": 6,
    },
}

# Atari 2600 has 128 bytes of RAM total, located from $80 to $FF
REAL_RAM_START = 0x80


def encode_bcd(score: int, byte_count: int, bcd_order: str = "msb", digit_order: str = "msb") -> list[int]:
    digits = [int(d) for d in f"{score:0{byte_count * 2}d}"]

    if digit_order == "lsb":
        digits = digits[::-1]

    bcd = []
    for i in range(0, len(digits), 2):
        hi, lo = digits[i], digits[i + 1]
        byte = (hi << 4) | lo
        bcd.append(byte)

    if bcd_order == "lsb":
        bcd = bcd[::-1]

    return bcd


def decode_score_bcd(ram, addr_list, config):
    base = REAL_RAM_START
    score_type = config["score_type"]
    bcd_order = config.get("bcd_order", "msb")
    digit_order = config.get("digit_order", "msb")
    multiplier = config.get("score_multiplier", 1)

    if score_type == "custom_battlezone":
        val1 = ram[0x9D - REAL_RAM_START]
        val2 = ram[0x9E - REAL_RAM_START]

        ones = (val1 >> 4) & 0xF  # left nibble of 0x9D
        tens = val2 & 0xF  # right nibble of 0x9E
        hundreds = (val2 >> 4) & 0xF  # left nibble of 0x9E

        for digit in (ones, tens, hundreds):
            if digit == 0xA:
                digit = 0  # unused

        score = (hundreds * 100 + tens * 10 + ones) * 1000
        return score

    if score_type == "custom_defender":
        # each address holds one digit, LSB first
        score = 0
        mult = 1
        for addr in addr_list:
            val = ram[addr - base] & 0xF
            if val == 0xA:
                val = 0
            score += val * mult
            mult *= 10
        return score * multiplier

    # decode BCD packed bytes
    ram_bytes = addr_list if bcd_order == "msb" else list(reversed(addr_list))
    digits = []
    for addr in ram_bytes:
        byte_val = ram[addr - base]
        digits.append((byte_val >> 4) & 0xF)  # left
        digits.append(byte_val & 0xF)  # right

    if digit_order == "lsb":
        digits = digits[::-1]

    score = sum(d * (10**i) for i, d in enumerate(reversed(digits)))
    return score * multiplier


def decode_lives(ram, addr, config):
    idx = addr - REAL_RAM_START
    # print(f"RAM[0x{addr:X}] = {ram[addr] - REAL_RAM_START]}")
    return ram[idx]


def write_score(config, score, ale):
    addr = config["score_addr"]
    score_type = config["score_type"]
    multiplier = config.get("score_multiplier", 1)

    if score_type == "packed_bcd":
        score = score // multiplier
        byte_count = len(addr)
        bcd = encode_bcd(score, byte_count, config.get("bcd_order", "msb"), config.get("digit_order", "msb"))

        for a, val in zip(addr, bcd):
            ale.setRAM(a - REAL_RAM_START, val)

    elif score_type == "custom_battlezone":
        digits = [int(d) for d in f"{score // multiplier:03d}"]  # 3-digit string
        hundreds, tens, ones = digits

        byte1 = ones << 4  # 0x9D = O0  (left nibble = ones)
        byte2 = (hundreds << 4) | tens  # 0x9E = HT

        ale.setRAM(0x9D - REAL_RAM_START, byte1)
        ale.setRAM(0x9E - REAL_RAM_START, byte2)

    elif score_type == "custom_defender":
        digits = [int(d) for d in f"{score:06d}"][::-1]  # reverse for LSB-first
        for a, d in zip(addr, digits):
            val = d & 0xF  # ensure upper nibble is 0
            ale.setRAM(a - REAL_RAM_START, val)

        # for addr in config["score_addr"]:
        #    print(f"RAM[{hex(addr)}] = {ale.getRAM()[addr - REAL_RAM_START]}")

    elif score_type == "nibble_lsb_first":
        # each digit is one nibble, LSB at lowest addr
        digits = [int(d) for d in f"{score:06d}"]
        for i, a in enumerate(addr):
            ale.setRAM(a - REAL_RAM_START, digits[i])

    else:
        raise NotImplementedError(f"Unsupported score_type: {score_type}")


def write_lives(config, lives, ale):
    addr = config["lives_addr"]
    idx = addr - REAL_RAM_START
    ram = ale.getRAM()
    total = config.get("total_lives", lives)
    display = config.get("display_lives", total)
    offset = config.get("lives_offset", 1 if display < total else 0)

    if config.get("lives_signed"):
        val = np.uint8(np.int8(lives - offset))
        ale.setRAM(idx, val)
        return

    val = (lives - offset) & 0xF

    nibble = config.get("lives_nibble", "low")
    current_val = ram[idx]

    if nibble == "high":
        new_val = (val << 4) | (current_val & 0x0F)
        ale.setRAM(idx, new_val)
    elif nibble == "low":
        new_val = (current_val & 0xF0) | val
        ale.setRAM(idx, new_val)
    else:
        raise ValueError(f"Unknown lives_nibble value: {nibble}")
