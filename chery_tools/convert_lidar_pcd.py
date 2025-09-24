import os
import numpy as np
import argparse


def parse_lidar_pcd_file(pcd_path):
    """
    头部信息示例:

    VERSION 0.7
    FIELDS x y z intensity timestamp ring
    SIZE 4 4 4 4 8 2
    TYPE F F F U F U
    COUNT 1 1 1 1 1 1
    WIDTH 182342
    HEIGHT 1
    VIEWPOINT 0.0 0.0 0.0 1.0 0.0 0.0 0.0
    POINTS 182342
    DATA binary
    """
    with open(pcd_path, "rb") as f:
        # 读取头部信息
        header = []
        while True:
            line = f.readline().decode("utf-8").strip()

            if line.startswith("DATA"):
                data_type = line.split()[1]
                if data_type != "binary":
                    raise ValueError("仅支持 binary 数据格式")
                break

            if line:
                header.append(line)

        # 解析头部信息
        def get_header_value(key):
            for h in header:
                if h.startswith(key):
                    return h.split()[1:]
            return None

        fields = get_header_value("FIELDS")
        sizes = [int(s) for s in get_header_value("SIZE")]
        types = get_header_value("TYPE")
        counts = [int(c) for c in get_header_value("COUNT")]
        points = int(get_header_value("POINTS")[0])

        # 构建 dtype
        dtype_list = []
        for i, field in enumerate(fields):
            count = counts[i]
            size = sizes[i]
            typ = types[i]

            if typ == "F":
                np_type = np.float32 if size == 4 else np.float64 if size == 8 else None
            elif typ == "U":
                np_type = (
                    np.uint8
                    if size == 1
                    else (
                        np.uint16
                        if size == 2
                        else (
                            np.uint32 if size == 4 else np.uint64 if size == 8 else None
                        )
                    )
                )
            elif typ == "I":
                np_type = (
                    np.int8
                    if size == 1
                    else (
                        np.int16
                        if size == 2
                        else np.int32 if size == 4 else np.int64 if size == 8 else None
                    )
                )
            else:
                raise ValueError(f"不支持的类型: {typ}")

            if np_type is None:
                raise ValueError(f"无效的大小 {size} 对于类型 {typ}")

            for j in range(count):
                field_name = f"{field}_{j}" if count > 1 else field
                dtype_list.append((field_name, np_type))

        dtype = np.dtype(dtype_list)

        # 读取数据
        data = np.fromfile(f, dtype=dtype, count=points)

        # x y z intensity timestamp ring
        with open(os.path.basename(pcd_path).replace(".pcd", ".csv"), "w") as csv_file:
            csv_file.write(",".join(fields) + "\n")
            for point in data:
                line = ",".join(str(point[field]) for field in fields)
                csv_file.write(line + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Parse LIDAR PCD file and convert to CSV.")
    parser.add_argument("--pcd_path", type=str, help="Path to the input PCD file.")
    args = parser.parse_args()

    parse_lidar_pcd_file(args.pcd_path)