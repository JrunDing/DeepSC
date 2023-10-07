"""This is used to evaluate the DeepSC model"""
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from models.transceiver import Transeiver, Mine
from utils.tools import SeqtoText, BleuScore, SNR_to_noise, Similarity
from utils.trainer import greedy_decode, train_mine_step
from dataset.dataloader import return_loader
from parameters import para_config

if __name__ == '__main__':
    # Set random seed
    tf.random.set_seed(5)
    # choose performance metrics
    test_metrics = True
    test_bleu = True
    test_sentence_sim = False
    test_mi = False
    runs = 1  # 10次结果取平均
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
    # Load dataset
    train_dataset, test_dataset = return_loader(args)
    # Define the model
    mine_net = Mine()  # 互信息
    net = Transeiver(args)  # 收发机
    # Load the model from the checkpoint path
    checkpoints = tf.train.Checkpoint(Transceiver=net)
    a = tf.train.latest_checkpoint(args.checkpoint_path)
    checkpoints.restore(a)
    if test_mi:
        # learning rate
        optim_mi = tf.keras.optimizers.Adam(lr=0.001)
        for snr in args.test_snr:
            n_std = SNR_to_noise(snr)
            for (batch, (inp, tar)) in enumerate(test_dataset):
                loss_mine = train_mine_step(inp, tar, net, mine_net, optim_mi, args.channel, n_std)
            print("SNR %f loss mine %f" % (snr, loss_mine.numpy()))

    if test_metrics:
        if test_sentence_sim:
            metrics = Similarity(args.bert_config_path, args.bert_checkpoint_path, args.bert_dict_path)
        elif test_bleu:
            metrics = BleuScore(1, 0, 0, 0)
        else:
            raise Exception('Must choose bleu score or sentence similarity')
        BLEU_score = []
        # Start the evaluation
        for snr in args.test_snr:
            n_std = SNR_to_noise(snr)  # SNR转标准差
            word, target_word = [], []
            score = 0
            for run in range(runs):
                for (batch, (inp, tar)) in enumerate(test_dataset):  # (64, 31)
                    # inp = tf.constant([[1, 9628, 21626, 20114, 8651, 9423,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])
                    # tar = tf.constant([[1, 9628, 21626, 20114, 8651, 9423,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])
                    preds = greedy_decode(args, inp, net, args.channel, n_std)  # 贪婪编解码
                    sentences = preds.cpu().numpy().tolist()  # (64, 31)
                    result_string = list(map(StoT.sequence_to_text, sentences))
                    word = word + result_string  # 解码句子
                    # print(word)
                    target_sent = tar.cpu().numpy().tolist()
                    result_string = list(map(StoT.sequence_to_text, target_sent))
                    target_word = target_word + result_string  # 目标句子
                    # print(target_word)

                score1 = metrics.compute_score(word, target_word)
                score1 = np.array(score1)
                score1 = np.mean(score1)
                score += score1
                # print(
                #     'Run: {}; Type: VAL; BLEU Score: {:.5f}'.format(
                #         run, score1
                #     )
                # )

            score = score/runs
            BLEU_score.append(score)
            print(
                'SNR: {}; Type: VAL; BLEU Score: {:.5f}'.format(
                    snr, score
                )
            )

    plt.figure(0)
    plt.plot(args.test_snr, BLEU_score, label='BLEU')
    plt.scatter(args.test_snr, BLEU_score)
    # plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))  # x轴只显示整数
    plt.title("BLEU vs. SNR[dB]")
    plt.xlabel("SNR[dB]")
    plt.ylabel("BLEU")
    plt.legend()
    plt.show()

