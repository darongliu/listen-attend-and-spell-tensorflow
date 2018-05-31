class Hyperparams:
    seq_len = 256 # must be the multiple of 8
    hidden_units = 256

    attention_hidden_units

    n_mels=80

    vocab #with S
    embed_size

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
    test_data = 'harvard_sentences.txt'
    max_duration = 10.0   #????

    # signal processing
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
    encoder_num_banks = 16
    decoder_num_banks = 8
    num_highwaynet_blocks = 4
    r = 5 # Reduction factor. Paper => 2, 3, 5
    dropout_rate = .5

    ## reference encoder
    ref_enc_filters = [32, 32, 64, 64, 128, 128]
    ref_enc_size = [3,3]
    ref_enc_strides = [2,2]
    #ref_enc_gru_size = 128
    ref_enc_gru_size = embed_size

    # ## style token layer
    # token_num = 10
    # token_emb_size = 256
    # num_heads = 8
    # multihead_attn_num_unit = 128
    # style_att_type = 'mlp_attention'
    # attn_normalize = True

    # training scheme
    num_epochs=300
    lr = 0.001 # Initial learning rate.
    logdir = "./logdir/vctk"
    logfile="./log"
    sampledir = 'samples'
    batch_size = 32