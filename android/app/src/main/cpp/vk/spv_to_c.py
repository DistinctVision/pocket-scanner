import struct
import sys

src, dst, name = sys.argv[1:]
data = open(src, "rb").read()
if len(data) % 4 != 0:
    raise SystemExit("spirv size is not a multiple of 4")
words = struct.unpack("<" + str(len(data) // 4) + "I", data)
with open(dst, "w", encoding="utf-8") as out:
    out.write("static const uint32_t %s[] = {\n" % name)
    for index, word in enumerate(words):
        out.write("0x%08x," % word)
        if index % 8 == 7:
            out.write("\n")
    out.write("\n};\n")
