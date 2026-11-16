import os
import shutil
from pathlib import Path

YANG_DIR = Path('baselines/yang2026').resolve()
INST = YANG_DIR / 'source_instrumented'

if INST.exists():
    shutil.rmtree(INST)
shutil.copytree(YANG_DIR / 'source', INST)

# 1. dead_counters.h and .cpp
with open(INST / 'commonlib/dead_counters.h', 'w') as f:
    f.write("""#pragma once
extern long long g_dead_1;
extern long long g_dead_2;
extern long long g_dead_3;
extern long long g_dead_4;
extern long long g_dead_5;
extern long long g_dead_6;
extern long long g_dead_7;
""")

with open(INST / 'commonlib/dead_counters.cpp', 'w') as f:
    f.write("""#include "dead_counters.h"
long long g_dead_1 = 0;
long long g_dead_2 = 0;
long long g_dead_3 = 0;
long long g_dead_4 = 0;
long long g_dead_5 = 0;
long long g_dead_6 = 0;
long long g_dead_7 = 0;
""")

# 2. global_arithmetic.cpp
p_ga = INST / 'commonlib/global_arithmetic.cpp'
ga = p_ga.read_text()
ga = '#include "dead_counters.h"\n' + ga
ga = ga.replace(
    'if (direction == 15 || direction == 23)\n\t\treturn 8;\n\treturn 0;',
    'if (direction == 15 || direction == 23)\n\t\treturn 8;\n\tg_dead_1++;\n\treturn 0;'
)
p_ga.write_text(ga)

# 3. utility.cpp
p_ut = INST / 'commonlib/utility.cpp'
ut = p_ut.read_text()
ut = '#include "dead_counters.h"\n' + ut
ut = ut.replace(
    '(*start_y) -= 1;\n\t\t\tdirection_list[(*num)++] = 1;\n\t\t\tbreak;\n\t\t}\n\t}\n\treturn 0;',
    '(*start_y) -= 1;\n\t\t\tdirection_list[(*num)++] = 1;\n\t\t\tbreak;\n\t\t}\n\t}\n\tg_dead_2++;\n\treturn 0;'
)
ut = ut.replace(
    'else if (direction == 25 || direction == 28 || direction == 31 ||\n\t\tdirection == 34)\n\t\treturn 8;\n\treturn 0;',
    'else if (direction == 25 || direction == 28 || direction == 31 ||\n\t\tdirection == 34)\n\t\treturn 8;\n\tg_dead_3++;\n\treturn 0;'
)
ut = ut.replace(
    '(index == 3 && direction == 26))\n\t\treturn 32;\n\treturn 0;',
    '(index == 3 && direction == 26))\n\t\treturn 32;\n\tg_dead_4++;\n\treturn 0;'
)
ut = ut.replace(
    'case -1:\n\t\treturn 20;\n\t}\n\treturn 0;',
    'case -1:\n\t\treturn 20;\n\t}\n\tg_dead_5++;\n\treturn 0;'
)
ut = ut.replace(
    'if (*turn == 1)\n\t\t\treturn 2;\n\t\telse\n\t\t{\n\t\t\t*turn = 1;\n\t\t\treturn 1;\n\t\t}\n\t}\n\treturn 0;',
    'if (*turn == 1)\n\t\t\treturn 2;\n\t\telse\n\t\t{\n\t\t\t*turn = 1;\n\t\t\treturn 1;\n\t\t}\n\t}\n\tg_dead_6++;\n\treturn 0;'
)
p_ut.write_text(ut)

# 4. decoder.cpp
p_dec = INST / 'decoder/decoder.cpp'
dec = p_dec.read_text()
dec = '#include "dead_counters.h"\n#include <chrono>\n' + dec
dec = dec.replace(
    'turn = 2;\n\t\t}\n\t}\n\treturn 0;',
    'turn = 2;\n\t\t}\n\t}\n\tg_dead_7++;\n\treturn 0;'
)
dec = dec.replace(
    'acodec.start_decoder();',
    'auto _t_start = std::chrono::steady_clock::now();\n\tacodec.start_decoder();'
)
dec = dec.replace(
    'acodec.stop_decoder();\n\n\tint offset = (pic_height * pic_width * (type + 2)) >> 1;',
    'acodec.stop_decoder();\n\tauto _t_end = std::chrono::steady_clock::now();\n\tdouble _core_ms = std::chrono::duration<double, std::milli>(_t_end - _t_start).count();\n\tstd::cout << "CHRONO_CORE_DECODE_MS: " << _core_ms << std::endl;\n\tstd::cout << "DEAD_COUNTERS_DEC: " << g_dead_7 << std::endl;\n\tstd::cerr << "DEAD_COUNTERS_DEC: " << g_dead_7 << std::endl;\n\tint offset = (pic_height * pic_width * (type + 2)) >> 1;'
)
p_dec.write_text(dec)

# 5. encoder.cpp
p_enc = INST / 'encoder/encoder.cpp'
enc = p_enc.read_text()
enc = '#include "dead_counters.h"\n#include <chrono>\n' + enc
enc = enc.replace('getchar();', 'return -1;')
enc = enc.replace(
    'encodeHead(buffer, rows, cols, frameNum, type, fileout, encodeStream, cMap,\n\t\t&numC);',
    'auto _t_start = std::chrono::steady_clock::now();\n\tencodeHead(buffer, rows, cols, frameNum, type, fileout, encodeStream, cMap, &numC);'
)
enc = enc.replace(
    'int bytenum = acodec.stop_encoder();',
    'int bytenum = acodec.stop_encoder();\n\tauto _t_end = std::chrono::steady_clock::now();\n\tdouble _core_ms = std::chrono::duration<double, std::milli>(_t_end - _t_start).count();\n\tstd::cout << "CHRONO_CORE_ENCODE_MS: " << _core_ms << std::endl;\n\tstd::cout << "DEAD_COUNTERS_ENC: " << g_dead_1 << " " << g_dead_2 << " " << g_dead_3 << " " << g_dead_4 << " " << g_dead_5 << " " << g_dead_6 << std::endl;\n\tstd::cerr << "DEAD_COUNTERS_ENC: " << g_dead_1 << " " << g_dead_2 << " " << g_dead_3 << " " << g_dead_4 << " " << g_dead_5 << " " << g_dead_6 << std::endl;'
)
p_enc.write_text(enc)

print("Instrumented source prepared successfully!")
