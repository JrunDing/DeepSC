"""
@Author: Jrun Ding
@Date: 2023.9.30
@Brief: 对原始的英文文本数据集进行处理，生成json字典、训练和测试数据集pkl文件，训练测试数据集中内容形式是
[[0, 23, 35, 1], [0, 25, 16, ……,1], ……]，即将每个句子按照字典编码的结果 原始数据集90%作为训练集
@Coding: utf-8
"""
import unicodedata
import re
import pickle
import argparse
import os
import json
from tqdm import tqdm
from w3lib.html import remove_tags

parser = argparse.ArgumentParser()
parser.add_argument('--input-data-dir', default='/en', type=str)
parser.add_argument('--output-train-dir', default='train_data.pkl', type=str)
parser.add_argument('--output-test-dir', default='test_data.pkl', type=str)
parser.add_argument('--output-vocab', default='vocab.json', type=str)

# 特殊字符串token
SPECIAL_TOKENS = {
    '<PAD>': 0,  # 补充
    '<START>': 1,  # 开始
    '<END>': 2,  # 结束
    '<UNK>': 3,  # 不知道的
}


def unicode_to_ascii(s):
    """
    :brief: 将unicode转为ascii编码形式的字符串
    :param s: 字符串句子s
    :return: ascii格式的字符串
    """
    return ''.join(c for c in unicodedata.normalize('NFD', s)  # normalize按NFD规则进行规则化
                   if unicodedata.category(c) != 'Mn')  # 判断字符串是否属于Mn形式


def normalize_string(s):
    """
    :brief: 按规则处理字符串
    :param s: 字符串句子s
    :return: 处理后的s
    """
    # 规则化unicode形式字符串，转为ascii形式
    s = unicode_to_ascii(s)
    # 去除XML文件的全部标签，只保留原始文本内容，得到纯净的每行字符串
    s = remove_tags(s)
    # 在!.?等一些指定的位置添加空格
    s = re.sub(r'([!.?])', r' \1', s)
    s = re.sub(r'[^a-zA-Z.!?]+', r' ', s)
    s = re.sub(r'\s+', r' ', s)
    # 全部改成小写
    s = s.lower()
    return s


def cutted_data(cleaned, MIN_LENGTH=4, MAX_LENGTH=30):
    """
    :brief: 只保留字符串列表中，长度大于4，小于30的元素，不满足的直接删掉
    :param cleaned: 原始的字符串列表
    :param MIN_LENGTH: 最小长度
    :param MAX_LENGTH: 最大长度
    :return: 处理后的字符串列表
    """
    cutted_lines = list()
    for line in cleaned:
        length = len(line.split())
        if MIN_LENGTH < length < MAX_LENGTH:
            line = [word for word in line.split()]
            cutted_lines.append(' '.join(line))
    return cutted_lines


def save_clean_sentences(sentence, save_path):
    pickle.dump(sentence, open(save_path, 'wb'))
    print('Saved: %s' % save_path)


def process(text_path):
    """
    :brief: 处理一个txt文本
    :param text_path: 文本路径
    :return: 满足规则要求的字符串列表
    """
    fop = open(text_path, 'r', encoding='utf8')
    raw_data = fop.read()
    sentences = raw_data.strip().split('\n')  # 移除开头结尾的" "，以'\n'分割每行生成列表
    raw_data_input = [normalize_string(data) for data in sentences]  # 规则化每行，返回每行列表，全部小写
    raw_data_input = cutted_data(raw_data_input)  # 删除不满足长度要求的列表字符串元素
    fop.close()

    return raw_data_input


def tokenize(s, delim=' ', add_start_token=True, add_end_token=True,
             punct_to_keep=None, punct_to_remove=None):
    """
    :brief: 分词器，将一个句子分割成单词，保留的标点等和单词同等地位，得到一个单词列表
    Tokenize a sequence, converting a string s into a list of (string) tokens by
    splitting on the specified delimiter. Optionally keep or remove certain
    punctuation marks and add start and end tokens.
    :param s: 句子字符串
    :param delim: 以什么分割句子得到单词，默认‘ ’
    :param add_start_token: 添加开始token
    :param add_end_token: 添加结束token
    :param punct_to_keep: 保留的punctuation
    :param punct_to_remove: 去除的punctuation
    :return: 一个单词列表
    """
    if punct_to_keep is not None:
        for p in punct_to_keep:
            s = s.replace(p, '%s%s' % (delim, p))  # 保留的内容前面加个空格

    if punct_to_remove is not None:
        for p in punct_to_remove:  # 去除的内容直接删掉
            s = s.replace(p, '')

    tokens = s.split(delim)  # 句子按空格分开得到单词、标点列表
    if add_start_token:
        tokens.insert(0, '<START>')
    if add_end_token:
        tokens.append('<END>')
    return tokens


def build_vocab(sequences, token_to_idx={}, min_token_count=1, delim=' ',
                punct_to_keep=None, punct_to_remove=None, ):
    """
    :brief: 将一个句子列表，生成一个统计全部单词并对每个单词进行编码并包含特殊<START>等的字典
    :param sequences: 句子字符串列表
    :param token_to_idx: 用于一些特殊token
    :param min_token_count: 最小token计数，数量小于它的单词则不统计
    :param delim: 以什么分割句子得到单词？
    :param punct_to_keep: 保留的标点符号
    :param punct_to_remove: 去除的标点符号
    :return: 全部句子的单词字典
    """
    token_to_count = {}  # 字典，用于统计单词

    # 对于每一个句子字符串
    for seq in sequences:
        seq_tokens = tokenize(seq, delim=delim, punct_to_keep=punct_to_keep,
                              punct_to_remove=punct_to_remove,
                              add_start_token=False, add_end_token=False)  # 对句子进行分词，得到单词列表
        for token in seq_tokens:
            if token not in token_to_count:  # 统计单词
                token_to_count[token] = 0  # 如果出现一次为0，两次为1……
            token_to_count[token] += 1

    # 以字典中count值排序，删去出现次数少的单词
    for token, count in sorted(token_to_count.items()):
        if count >= min_token_count:
            token_to_idx[token] = len(token_to_idx)  # 对单词设置编码规则，以之前的special_token字典继续添加单词

    return token_to_idx


def encode(seq_tokens, token_to_idx, allow_unk=False):
    seq_idx = []
    for token in seq_tokens:
        if token not in token_to_idx:
            if allow_unk:
                token = '<UNK>'
            else:
                raise KeyError('Token "%s" not in vocab' % token)
        seq_idx.append(token_to_idx[token])
    return seq_idx


def decode(seq_idx, idx_to_token, delim=None, stop_at_end=True):
    tokens = []
    for idx in seq_idx:
        tokens.append(idx_to_token[idx])
        if stop_at_end and tokens[-1] == '<END>':
            break
    if delim is None:
        return tokens
    else:
        return delim.join(tokens)


def main(args):
    data_dir = '../data_test/'
    args.input_data_dir = data_dir + args.input_data_dir
    args.output_train_dir = data_dir + args.output_train_dir
    args.output_test_dir = data_dir + args.output_test_dir
    args.output_vocab = data_dir + args.output_vocab

    print(args.input_data_dir)
    sentences = []
    print('Preprocess Raw Text')
    # for循环处理路径下的全部txt文件，把所有满足规则的句子合并成一个列表，并进行全部小写等一系列处理
    for fn in tqdm(os.listdir(args.input_data_dir)):
        if not fn.endswith('.txt'): continue  # 如果不是以.txt结尾则查看下一个
        process_sentences = process(os.path.join(args.input_data_dir, fn))  # 处理每个txt文件
        sentences += process_sentences

    # 去掉相同句子，得到一个字典a，键是句子字符串，值是该句子字符串出现的次数
    a = {}
    for set in sentences:
        if set not in a:
            a[set] = 0
        a[set] += 1  # 统计每个句子出现的次数，一次为0，两次为1
    sentences = list(a.keys())  # sentences还是全部句子的字符串列表
    print('Number of sentences: {}'.format(len(sentences)))  # 统计不重复句子的数量

    # 得到单词到索引的字典 {'<START>':0, ……, 'fuck':100,……}
    print('Build Vocab')
    token_to_idx = build_vocab(
        sentences, SPECIAL_TOKENS,
        punct_to_keep=[';', ','], punct_to_remove=['?', '.']
    )
    vocab = {'token_to_idx': token_to_idx}
    print('Number of words in Vocab: {}'.format(len(token_to_idx)))

    # 保存词典到json
    if args.output_vocab != '':
        with open(args.output_vocab, 'w') as f:
            json.dump(vocab, f)

    # 编码
    print('Start encoding txt')
    results = []  # 保留编码后的整数列表 [[0, 23, 35, 1], [0, 25, 16, ……,1], ……]  二维中每个列表都是一个句子
    count_len = []  # 用于统计每个句子中的单词数量 [23, 15, ……]
    for seq in tqdm(sentences):  # 对于每一个句子
        words = tokenize(seq, punct_to_keep=[';', ','], punct_to_remove=['?', '.'])  # 分词
        tokens = [token_to_idx[word] for word in words]  # 根据词典编码
        count_len.append(len(tokens))  # 统计每个句子的单词数量
        results.append(tokens)  # 编码后整数二维列表

    print('Writing Data')
    train_data = results[: round(len(results) * 0.1)]  # 全部数据的90%作为训练数据[[0, 23, 35, 1], [0, 25, 16, ……,1], ……]
    test_data = results[round(len(results) * 0.9):]  # 10%作为测试数据[[0, 23, 35, 1], [0, 25, 16, ……,1], ……]

    with open(args.output_train_dir, 'wb') as f:  # 把[[0, 23, 35, 1], [0, 25, 16, ……,1], ……]直接dump进pickle文件中
        pickle.dump(train_data, f)
    with open(args.output_test_dir, 'wb') as f:  # 把[[0, 23, 35, 1], [0, 25, 16, ……,1], ……]直接dump进pickle文件中
        pickle.dump(test_data, f)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
