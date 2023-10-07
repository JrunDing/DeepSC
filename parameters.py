
import argparse


def para_config():
    parser = argparse.ArgumentParser()

    # preprocessing parameters
    parser.add_argument('--input-data-dir', default='europarl/txt/en', type=str)
    parser.add_argument('--output-train-dir', default='europarl/train_data.pkl', type=str)
    parser.add_argument('--output-test-dir', default='europarl/test_data.pkl', type=str)
    parser.add_argument('--output-vocab', default='europarl/vocab.json', type=str)

    parser.add_argument('--train-save-path', default='data/europarl/train_data.pkl', type=str)
    parser.add_argument('--test-save-path', default='data/europarl/test_data.pkl', type=str)
    parser.add_argument('--vocab-path', default='data/europarl/vocab.json', type=str)

    # Training parameters
    parser.add_argument('--bs', default=1, type=int, help='The training batch size')
    parser.add_argument('--shuffle-size', default=2000, type=int, help='The training shuffle size')
    parser.add_argument('--lr', default=5e-4, type=float, help='The training learning rate')
    parser.add_argument('--epochs', default=60, type=int, help='The training number of epochs')
    parser.add_argument('--train-with-mine',  default=False, action='store_true',
                    help='If added, the network will be trained WITH Mutual Information')
    parser.add_argument('--checkpoint-path', default='./checkpoints/', type=str,
                        help='The path to save model')
    parser.add_argument('--max-length', default=40, type=int, help='Max length of sentence')
    parser.add_argument('--channel', default='AWGN', type=str, help='Choose the channel to simulate')

    # Model parameters
    parser.add_argument('--encoder-num-layer', default=4, type=int, help='The number of encoder layers')
    parser.add_argument('--encoder-d-model', default=128, type=int, help='The output dimension of attention')
    parser.add_argument('--encoder-d-ff', default=512, type=int, help='The output dimension of ffn')
    parser.add_argument('--encoder-num-heads', default=8, type=int, help='The number heads')
    parser.add_argument('--encoder-dropout', default=0.1, type=float, help='The encoder dropout rate')

    parser.add_argument('--decoder-num-layer', default=4, type=int, help='The number of decoder layers')
    parser.add_argument('--decoder-d-model', default=128, type=int, help='The output dimension of decoder')
    parser.add_argument('--decoder-d-ff', default=512, type=int, help='The output dimension of ffn')
    parser.add_argument('--decoder-num-heads', default=8, type=int, help='The number heads')
    parser.add_argument('--decoder-dropout', default=0.1, type=float, help='The decoder dropout rate')

    # Other parameter settings
    parser.add_argument('--train-snr', default=6, type=int, help='The train SNR')
    parser.add_argument('--test-snr', default=[-10, -5, 0, 5, 10, 15, 20], type=list, help='The test SNR')
    # Mutual Information Model Parameters

    args = parser.parse_args()

    return args

