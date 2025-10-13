import os
import numpy as np
import argparse


def parse_lidar_pcd_file(pcd_path, return_fields=False):
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

        if return_fields:
            return data, fields
        return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将 LIDAR PCD 文件转换为 CSV 文件")
    parser.add_argument("--path", type=str, help="Path to the input PCD file.")
    args = parser.parse_args()

    if args.path.endswith(".pcd"):
        parse_func = parse_lidar_pcd_file
        
    data, fields = parse_func(args.path, return_fields=True)
    
    save_path = os.path.basename(args.path).replace(".pcd", ".csv")
    with open(save_path, "w") as csv_file:
        csv_file.write(",".join(fields) + "\n")
        for point in data:
            if point.ndim == 0:
                line = ",".join(str(point[field]) for field in fields)
                csv_file.write(line + "\n")
            else:
                line = ",".join(str(value) for value in point)
                csv_file.write(line + "\n")
                
