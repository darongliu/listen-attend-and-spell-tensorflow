'''
modified from
https://www.github.com/kyubyong/tacotron
'''
from hyperparams import Hyperparams as hp
import tqdm 
from data_load import load_data, load_vocab
import tensorflow as tf
from graph import Graph
import os
import numpy as np

def load_pre_spectrograms(fpath):
    fname = os.path.basename(fpath)
    mel = hp.prepro_path + "/mels/{}".format(fname.replace("wav", "npy"))
    mag = hp.prepro_path + "/mags/{}".format(fname.replace("wav", "npy"))
    mel = np.load(mel)
    mel = np.reshape(mel, [-1, hp.n_mels])
    t = mel.shape[0]
    num_paddings = 8 - (t % 8) if t % 8 != 0 else 0
    mel = np.pad(mel, [[0, num_paddings], [0, 0]], mode="constant")
    return fname, mel, np.load(mag)

def get_sent(idx2char, idx_np):
    all_sent = []
    for i in range(len(idx_np)):

def wer(w,h):
    pass

def evaluate():
    # Load graph
    g = Graph(mode="evaluate"); print("Graph loaded")

    # Load data
    _, idx2char = load_vocab()
    fpaths, _, texts = load_data(mode="evaluate")
    all_mel_spec = [load_pre_spectrograms(fpath)[1] for fpath in fpaths]
    lengths = max([len(m) for m in all_mel_spec])
    new_mel_spec = np.zeros((len(all_mel_spec), maxlen, hp.n_mels), np.float)
    for i, m_spec in enumerate(all_mel_spec):
        new_mel_spec[i, :len(m_spec), :] = m_spec

    saver = tf.train.Saver()
    opf = open("Inference_text_seqs.txt", "w") #inference output
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint(hp.logdir)); print("Evaluate Model Restored!")
        y_hat = np.zeros((len(texts), 100), np.float32)
        for j in tqdm.tqdm(range(100)):
            _y_hat = sess.run(g.y_hat, {g.x: new_mel_spec, g.y: y_hat})
            y_hat[:, j] = _y_hat[:, j]

        for i, text in enumerate(y_hat):
            fname = all_feat[i][0]
            text_gt = texts[i]
            final_str = fname + ","
            for t in text_gt:
                final_str = final_str + idx2char[t]
            final_str = final_str + ","
            for t in text:
                final_str = final_str + idx2char[t]
            final_str = final_str + "\n"

            opf.write(final_str)
if __name__ == '__main__':
    evaluate()
    print("Done")