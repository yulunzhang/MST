import tensorflow as tf

def wct(content, style, num_feature=512, epsilon=1e-8, style_ratio=1):
    content_t = tf.squeeze(content, axis=0)
    style_t   = tf.squeeze(style, axis=0)

    Cc, Hc, Wc = tf.shape(content_t)[0], tf.shape(content_t)[1], tf.shape(content_t)[2]
    Cs, Hs, Ws = tf.shape(style_t)[0], tf.shape(style_t)[1], tf.shape(style_t)[2]

    ## C x H x W --> C x (HxW)
    content_flat = tf.reshape(content_t, (Cc, Hc*Wc))
    style_flat   = tf.reshape(style_t, (Cs, Hs*Ws))
    ## content mean in each channel
    mc = tf.reduce_mean(content_flat, axis=1, keep_dims=True)
    fc = content_flat - mc
    fc_conv = tf.matmul(fc, fc, transpose_b=True) / tf.cast(Hc*Wc - 1, tf.float32) + tf.eye(Cc) * epsilon
    
    ms = tf.reduce_mean(style_flat, axis=1, keep_dims=True)
    fs = style_flat - ms
    fs_conv = tf.matmul(fs, fs, transpose_b=True) / tf.cast(Hs*Ws - 1, tf.float32) + tf.eye(Cs) * epsilon

    with tf.device('/cpu:0'):
        Sc, Uc, _ = tf.svd(fc_conv)
        Ss, Us, _ = tf.svd(fs_conv)
    
    k_c = tf.reduce_sum(tf.cast(tf.greater(Sc, 1e-5), tf.int32))
    k_s = tf.reduce_sum(tf.cast(tf.greater(Ss, 1e-5), tf.int32))

    Dc = tf.diag(tf.pow(Sc[:k_c], -0.5))
    fc_hat = tf.matmul(tf.matmul(tf.matmul(Uc[:, :k_c], Dc), Uc[:, :k_c], transpose_b=True), fc)

    Ds = tf.diag(tf.pow(Ss[:k_s], 0.5))
    fcs_hat = tf.matmul(tf.matmul(tf.matmul(Us[:, :k_s], Ds), Us[:, :k_s], transpose_b=True), fc_hat)
    fcs_hat = fcs_hat + ms

    fcs_hat_mix = style_ratio * fcs_hat + (1-style_ratio) * (fc + mc)
    fcs_hat_mix = tf.reshape(fcs_hat_mix, (num_feature, Hc, Wc))
    fcs_hat_mix = tf.expand_dims(fcs_hat_mix, 0)

    return fcs_hat_mix