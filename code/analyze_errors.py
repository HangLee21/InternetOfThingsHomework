from demodulate import demodulate_audio
from utils import rsencode, barray2binarray


def read_first_line(file_path):
    """读取指定文件的第一行文本"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            first_line = file.readline().strip()  # 读取第一行并去除首尾空白
            return first_line
    except FileNotFoundError:
        print(f"文件未找到: {file_path}")
        return None
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return None

def get_baseline_text(file_path = 'test/analyze_error_text.txt'):
    """获取基准文本"""
    first_line = read_first_line(file_path)
    if first_line is not None:
        print("第一行文本:", first_line)
    return first_line

def replace_invalid_characters(text):
    """替换非法字符为 \0"""
    if text is None:
        return None
    # 假设非法字符是 Unicode 无法编码的字符
    return ''.join(c if c.isprintable() else '\0' for c in text)

def analyze_bit_error_rate(baseline_text, decoded_text):
    """比较两个文本的比特错误"""

    # 编码基准文本和解码文本
    baseline_binary = rsencode(baseline_text)
    baseline_array = barray2binarray(baseline_binary)

    decoded_binary = rsencode(replace_invalid_characters(decoded_text))
    decoded_array = barray2binarray(decoded_binary)

    # 计算比特错误
    # 确保两个数组长度相同
    min_length = min(len(baseline_array), len(decoded_array))
    bit_errors = sum(1 for i in range(min_length) if baseline_array[i] != decoded_array[i])

    # 返回比特错误数量和总比特数
    return bit_errors / min_length, bit_errors, min_length

def analyze_symbol_error_rate(baseline_text, decoded_text):
    """分析符号错误率 (SER)"""

    # 将文本视为符号序列
    baseline_symbols = list(baseline_text)
    decoded_symbols = list(decoded_text)

    # 确保符号长度相同
    min_length = min(len(baseline_symbols), len(decoded_symbols))

    # 计算符号错误
    symbol_errors = sum(1 for i in range(min_length) if baseline_symbols[i] != decoded_symbols[i])

    # 计算符号错误率
    if min_length == 0:
        return 1  # 避免除以零
    ser = symbol_errors / min_length

    return ser, symbol_errors, min_length

def main():
    baseline_text = get_baseline_text()
    decoded_text = demodulate_audio("output/record.wav")

    print(f'baseline:\t{baseline_text}')
    print(f'decoded:\t{decoded_text}')
    ber = analyze_bit_error_rate(baseline_text, decoded_text)
    ser = analyze_symbol_error_rate(baseline_text, decoded_text)
    print(f'SER: {ser}, BER: {ber}')



main()


