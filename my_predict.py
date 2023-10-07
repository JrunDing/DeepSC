import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from models.transceiver import Transeiver, Mine
from utils.tools import SeqtoText, BleuScore, SNR_to_noise
from utils.trainer import greedy_decode
from utils.sentence_sim import Sentence_sim
from parameters import para_config
from dataset.preprocess_text import tokenize
from w3lib.html import remove_tags


bleu_huffman_list = [0.0, 0.0, 0.0469524, 0.235229, 0.953768, 0.953768, 0.953768]
sim_huffman_list = [0.005208, 0.02078, 0.057684, 0.138788, 0.939098, 0.939098, 0.939098]
bleu_5bit_list = [0.0, 0.0, 0.0, 0.068187, 1.0, 1.0, 1.0]
sim_5bit_list = [0.030084, 0.0048839, 0.007728, 0.0172939, 1.0, 1.0, 1.0]


if __name__ == '__main__':
    # Set random seed
    tf.random.set_seed(5)
    # choose performance metrics
    test_mi = False
    # Set Parameters
    args = para_config()
    # Load the vocab
    vocab = json.load(open(args.vocab_path, 'rb'))
    args.vocab_size = len(vocab['token_to_idx'])
    token_to_idx = vocab['token_to_idx']  # 词典
    args.pad_idx = token_to_idx["<PAD>"]
    args.start_idx = token_to_idx["<START>"]
    args.end_idx = token_to_idx["<END>"]
    StoT = SeqtoText(token_to_idx, args.end_idx)

    # Define the model
    mine_net = Mine()  # 互信息
    net = Transeiver(args)  # 收发机
    # Load the model from the checkpoint path
    checkpoints = tf.train.Checkpoint(Transceiver=net)
    a = tf.train.latest_checkpoint(args.checkpoint_path)
    checkpoints.restore(a)
    '''
    if test_mi:
        # learning rate
        optim_mi = tf.keras.optimizers.Adam(lr=0.001)
        for snr in args.test_snr:
            n_std = SNR_to_noise(snr)
            for (batch, (inp, tar)) in enumerate(test_dataset):
                loss_mine = train_mine_step(inp, tar, net, mine_net, optim_mi, args.channel, n_std)
            print("SNR %f loss mine %f" % (snr, loss_mine.numpy()))
    '''

    test_sentences = ['i understand what you are saying',
                      'it is already clear that parliament is going to evaluate the '
                      'first and second reports differently',
                      'that is the position which a clear majority of the committee on budgetary control subscribes to',
                      'we need tough regulations which can be implemented',
                      'i want to emphasise strongly that changing our control systems will definitely not mean '
                      'relaxing them']
    # 编码
    encoded_test_sentences = []
    for seq in test_sentences:  # 对于每一个句子
        words = tokenize(seq, punct_to_keep=[';', ','], punct_to_remove=['?', '.'])  # 分词
        tokens = [token_to_idx[word] for word in words]  # 根据词典编码
        encoded_test_sentences.append(tokens)  # 编码后整数二维列表
    # 评估指标
    metrics_sim = Sentence_sim()
    metrics_bleu = BleuScore(1, 0, 0, 0)

    BLEU_score = []
    sentence_sim_score = []
    for snr in args.test_snr:
        n_std = SNR_to_noise(snr)  # SNR转标准差
        word, target_word = [], []
        score = 0
        score_sim = 0
        for encoded_test_sentence in encoded_test_sentences:
            inp = tf.expand_dims(tf.constant(encoded_test_sentence), axis=0)
            tar = inp
            # 网络
            preds = greedy_decode(args, inp, net, args.channel, n_std)  # 贪婪编解码
            sentences = preds.cpu().numpy().tolist()  # (64, 31)
            result_string = list(map(StoT.sequence_to_text, sentences))
            word = word + result_string  # 解码句子
            # print(word)
            # 实际
            target_sent = tar.cpu().numpy().tolist()
            result_string = list(map(StoT.sequence_to_text, target_sent))
            target_word = target_word + result_string  # 目标句子
            # print(target_word)

        score = metrics_bleu.compute_score(word, target_word)
        score = np.array(score)
        score = np.mean(score)
        BLEU_score.append(score)

        sim_score_sum = 0
        for i in range(len(target_word)):
            sent1 = remove_tags(word[i]).strip()
            sent2 = remove_tags(target_word[i]).strip()
            score2 = metrics_sim.cal_sentence_sim(sent1, sent2)
            sim_score_sum += score2
        score_sim = sim_score_sum/len(target_word)
        sentence_sim_score.append(score_sim)

        print(
            'SNR: {}; Type: VAL; BLEU Score: {:.5f}'.format(
                snr, score
            )
        )
        print(
            'SNR: {}; Type: VAL; Sentence_sim Score: {:.5f}'.format(
                snr, score_sim
            )
        )

    plt.figure(0)
    plt.plot(args.test_snr, BLEU_score, label='DeepSC_BLEU')
    plt.plot(args.test_snr, sentence_sim_score, label='DeepSC_SS')
    plt.plot(args.test_snr, bleu_huffman_list, label='Huff_RS_BLEU')
    plt.plot(args.test_snr, sim_huffman_list, label='Huff_RS_SS')
    plt.plot(args.test_snr, bleu_5bit_list, label='5bit_RS_BLEU')
    plt.plot(args.test_snr, sim_5bit_list, label='5bit_RS_SS')

    plt.scatter(args.test_snr, BLEU_score)
    plt.scatter(args.test_snr, sentence_sim_score)
    plt.scatter(args.test_snr, bleu_huffman_list)
    plt.scatter(args.test_snr, sim_huffman_list)
    plt.scatter(args.test_snr, bleu_5bit_list)
    plt.scatter(args.test_snr, sim_5bit_list)
    plt.title("Metrics")
    plt.xlabel("SNR[dB]")
    plt.ylabel("Score")
    plt.legend()
    plt.show()

