class Hyperparams:
    '''Hyper parameters'''

    # pipeline
    prepro = True  # if True, run `python prepro.py` first before running `python train.py`.

    vocab = "PES abcdefghijklmnopqrstuvwxyz'.?" # P: Padding E: End of Sentence S: Start token
    #vocab = "PE abcdefghijklmnopqrstuvwxyzáéíóúüñ.?¿'!" #for spain

    # data
    prepro_path = "/home/givebirthday/data/VCTK/prepro_data"
    data = "/home/givebirthday/data/VCTK"
    # data = "/data/private/voice/nick"
    test_dir = ''

    # signal processing  # for preprocessing and tacotron
    sr = 16000 # Sample rate.
    n_fft = 2048 # fft points (samples)
    frame_shift = 0.0125 # seconds
    frame_length = 0.05 # seconds
    hop_length = int(sr*frame_shift) # samples.
    win_length = int(sr*frame_length) # samples.
    n_mels = 80 # Number of Mel banks to generate
    power = 1.2 # Exponent for amplifying the predicted magnitude
    n_iter = 300 # Number of inversion iterations #griffin-lim
    preemphasis = .97 # or None
    max_db = 100
    ref_db = 20

    # model
    ## tacotron-1
    embed_size = 256 # alias = E
    hidden_units = 512
    attention_hidden_units = 512
    r = 5 # Reduction factor. Paper => 2, 3, 5
    dropout_rate = .5

    # training scheme
    num_epochs=300
    lr = 0.001 # Initial learning rate.
    logdir = "./logdir/vctk"
    logfile="./log"
    batch_size = 32