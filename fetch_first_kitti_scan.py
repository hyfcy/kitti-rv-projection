# -*- coding: utf-8 -*-
"""
从 KITTI 官方 zip 的开头流式取出**第一个** velodyne 点云文件。

为什么可行：KITTI 的 zip 是 stored（无压缩）存储，且条目顺序就是把 00 序列
的 velodyne 放在最前面。所以顺序扫描开头的几 MB，就能拿到 000000.bin。

为什么不用解析中央目录：该压缩包约 81 GB，中央目录放在最末尾且很大，
读取尾部索引反而比读开头贵得多。

用法:
    python fetch_first_kitti_scan.py                     # 取 00/velodyne/000000.bin
    python fetch_first_kitti_scan.py --fetch-mb 16
    python fetch_first_kitti_scan.py --name 000001.bin
"""

import argparse
import os
import struct
import sys
import urllib.request

URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_velodyne.zip"
SIG_LOCAL = b"PK\x03\x04"
SIG_DATA_DESC = b"PK\x07\x08"


def opener(proxy):
    handlers = []
    if proxy:
        handlers.append(urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
    return urllib.request.build_opener(*handlers)


def get_range(url, proxy, start, end, timeout=600):
    req = urllib.request.Request(url)
    req.add_header("Range", "bytes=%d-%d" % (start, end))
    with opener(proxy).open(req, timeout=timeout) as r:
        return r.read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=URL)
    ap.add_argument("--proxy", default=os.environ.get("KITTI_PROXY", ""))
    ap.add_argument("--fetch-mb", type=int, default=12, help="先下载开头多少 MB")
    ap.add_argument("--name", default="000000.bin", help="想要的条目 basename")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = args.out or args.name
    if args.proxy:
        print("[i] 代理: %s" % args.proxy)

    n = args.fetch_mb * 1024 * 1024
    print("[i] 下载开头 %.0f MB ..." % (n / 1024 / 1024))
    buf = get_range(args.url, args.proxy, 0, n - 1)
    print("[i] 实得 %.2f MB" % (len(buf) / 1024 / 1024))

    # 顺序扫描本地文件头，解析出每个条目
    off = 0
    found = None
    scanned = 0
    while True:
        i = buf.find(SIG_LOCAL, off)
        if i < 0:
            break
        if i + 30 > len(buf):
            break
        (_, _, _, method, _, _, _, comp_size, uncomp_size,
         name_len, extra_len) = struct.unpack("<IHHHHHIIIHH", buf[i:i + 30])
        name = buf[i + 30: i + 30 + name_len].decode("utf-8", "replace")
        data_off = i + 30 + name_len + extra_len
        scanned += 1

        if comp_size == 0 and not name.endswith("/"):
            # 用了数据描述符：数据后面跟 PK\x07\x08 + sizes
            d = buf.find(SIG_DATA_DESC, data_off)
            if d < 0:
                print("[!] %s 需要数据描述符但未在缓冲区内找到，扩大 --fetch-mb" % name)
                break
            comp_size = struct.unpack("<I", buf[d + 8:d + 12])[0]
            uncomp_size = struct.unpack("<I", buf[d + 12:d + 16])[0]

        if name.endswith("/") or comp_size == 0:
            off = i + 4
            continue

        print("    [%d] %-40s %8.2f MB  stored=%s" %
              (scanned, name, uncomp_size / 1024 / 1024, method == 0))

        if os.path.basename(name) == args.name:
            if data_off + comp_size > len(buf):
                print("[!] 该条目需要 %.2f MB，但只下了 %.2f MB → 请增大 --fetch-mb"
                      % ((data_off + comp_size) / 1024 / 1024, len(buf) / 1024 / 1024))
                sys.exit(2)
            found = (name, data_off, comp_size, method)
            break

        off = data_off + comp_size

    if not found:
        sys.exit("没在开头 %d MB 内找到 %s，试试增大 --fetch-mb" % (args.fetch_mb, args.name))

    name, data_off, comp_size, method = found
    if method != 0:
        sys.exit("条目是压缩存储，本脚本只支持 stored（method=0）")

    blob = buf[data_off: data_off + comp_size]
    with open(out, "wb") as f:
        f.write(blob)
    print("[OK] 已写出 %s（%.2f MB）← %s" % (out, len(blob) / 1024 / 1024, name))
    print("     下一步: python rv_projection.py %s" % out)


if __name__ == "__main__":
    main()
