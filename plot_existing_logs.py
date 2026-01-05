import argparse
import os
import sys

# 添加当前目录到 sys.path 以便导入 utils
sys.path.append(os.getcwd())

from utils.plot_logs import generate_plots_from_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="根据已有的日志文件绘制学术风格的训练曲线图。")
    parser.add_argument("log_file", type=str, help="日志 JSON 文件的路径 (例如: sample_logs/sample_train_logs/log_data_2025-10-20_10-11-10.json)")
    parser.add_argument("--output_dir", type=str, default=None, help="保存图表的目录。默认保存在日志文件所在的目录。")
    parser.add_argument("--smoothing", type=float, default=0.9, help="曲线平滑系数 (0.0 - 1.0)，值越大越平滑。默认 0.9。")

    args = parser.parse_args()

    if not os.path.exists(args.log_file):
        print(f"❌ 错误: 找不到文件 '{args.log_file}'")
        sys.exit(1)

    print(f"📊 正在处理日志文件: {args.log_file} ...")
    try:
        generate_plots_from_file(args.log_file, args.output_dir, args.smoothing)
        print("✨ 绘图完成！")
    except Exception as e:
        print(f"❌ 绘图过程中发生错误: {e}")
