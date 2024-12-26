import argparse
from src.algorithms.API import LaplacianPyramid
import cv2

def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='图像对齐和堆叠工具')
    parser.add_argument('images', nargs='+', help='输入图像文件路径列表')
    parser.add_argument('--output', '-o', default='output.jpg', help='输出图像路径')
    parser.add_argument('--kernel-size', '-k', type=int, default=6, help='融合核大小')
    parser.add_argument('--pyramid-levels', '-p', type=int, default=8, help='金字塔层数')
    
    args = parser.parse_args()
    
    # 创建处理实例
    processor = LaplacianPyramid(
        fusion_kernel_size=args.kernel_size,
        pyramid_num_levels=args.pyramid_levels
    )
    
    # 设置图像路径
    processor.update_image_paths(args.images)
    
    # 处理图像
    result = processor.align_and_stack_images()
    
    # 保存结果
    cv2.imwrite(args.output, result)
    print(f"处理完成,结果已保存至: {args.output}")

if __name__ == "__main__":
    main() 