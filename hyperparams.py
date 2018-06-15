class Hyperparams:
    '''Hyper parameters'''

    # pipeline
    prepro = True  # if True, run `python prepro.py` first before running `python train.py`.

    vocab = "PES abcdefghijklmnopqrstuvwxyz'.?" # P: Padding E: End of Sentence S: Start token
    #vocab = "PE abcdefghijklmnopqrstuvwxyzáéíóúüñ.?¿'!" #for spain

    # data
    prepro_path = "/home/darong/darong/data/VCTK/prepro_data"
    data = "/home/darong/darong/data/VCTK"
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
    #tacotron
    r = 5 # Reduction factor. Paper => 2, 3, 5
    #las
    embed_size = 256
    hidden_units = 512
    attention_hidden_units = 512
    dropout_rate = 0.2 #rate=0.1 would drop out 10% of input units.
    attention_mechanism='original' #original #dot

    # training scheme
    num_epochs=100
    lr = 0.001 # Initial learning rate.
    #lr_decay=0.9 #decay whenever loss is larger than previous epoch
    logdir = "./logdir/vctk_dr_0.2_original"
    logfile="log"
    batch_size = 32

    # for inference
    inference_batch_size = 300 #how many batch per time
