'''
modified from
https://www.github.com/kyubyong/tacotron
'''
from hyperparams import Hyperparams as hp
import tqdm
from data_load import load_data
import tensorflow as tf
from graph import Graph
import os
import numpy as np
from utils import load_spectrograms

def wer(list1, list2):
    
def load_prepre_spectrograms(fpath):
    fname = os.path.basename(fpath.decode())
    mel = hp.prepro_path + "/mels/{}".format(fname.replace("wav", "npy"))
    mag = hp.prepro_path + "/mags/{}".format(fname.replace("wav", "npy"))
    mel = np.load(mel)
    mel = np.reshape(mel, [-1, hp.n_mels])
    t = mel.shape[0]
    num_paddings = 8 - (t % 8) if t % 8 != 0 else 0
    mel = np.pad(mel, [[0, num_paddings], [0, 0]], mode="constant")
    return fname, mel, np.load(mag)

def evaluate():
    # Load graph
    g = Graph(mode="evaluate"); print("Graph loaded")

    # Load data
    texts = load_data(mode="evaluate")
        

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint(hp.logdir)); print("Evaluate Model Restored!")
        """
        err = 0.0

        for i, t_split in enumerate(new_texts):
            y_hat = np.zeros((t_split.shape[0], 200, hp.n_mels*hp.r), np.float32)  # hp.n_mels*hp.r
            for j in tqdm.tqdm(range(200)):
                _y_hat = sess.run(g.y_hat, {g.x: t_split, g.y: y_hat})
                y_hat[:, j, :] = _y_hat[:, j, :]

            mags = sess.run(g.z_hat, {g.y_hat: y_hat})
            for k, mag in enumerate(mags):
                fname, mel_ans, mag_ans = load_spectrograms(new_fpaths[i][k])
                print("File {} is being evaluated ...".format(fname))
                audio = spectrogram2wav(mag)
                audio_ans = spectrogram2wav(mag_ans)
                err += calculate_mse(audio, audio_ans)

        err = err/float(len(fpaths))
        print(err)

        """
        # Feed Forward
        ## mel
        y_hat = np.zeros((new_texts.shape[0], 200, hp.n_mels*hp.r), np.float32)  # hp.n_mels*hp.r
        for j in tqdm.tqdm(range(200)):
            _y_hat = sess.run(g.y_hat, {g.x: new_texts, g.y: y_hat})
            y_hat[:, j, :] = _y_hat[:, j, :]
        ## mag
        mags = sess.run(g.z_hat, {g.y_hat: y_hat})
        err = 0.0
        for i, mag in enumerate(mags):
            fname, mel_ans, mag_ans = load_spectrograms(fpaths[i])
            print("File {} is being evaluated ...".format(fname))
            audio = spectrogram2wav(mag)
            audio_ans = spectrogram2wav(mag_ans)
            err += calculate_mse(audio, audio_ans)
        err = err/float(len(fpaths))
        print(err)
        opf.write(hp.logdir  + "  " + str(err) + "\n")

if __name__ == '__main__':
    evaluate()
    print("Done")

