#!/usr/bin/env python
from itertools import product
import argparse
import os
import time
import numpy as np
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import logging
import maxflow
from maxflow import fastmin

from src.image import load_image, prepare_image, save_image
from src.nn import build_vgg, build_decoder
from src.norm import wct
from src.weights import open_weights
from src.util import get_filename, get_params, extract_image_names_recursive

def style_transfer(
        content=None,
        content_dir=None,
        content_size=512,
        style=None,
        style_dir=None,
        style_size=512,
        crop=None,
        alpha=1.0,
        output_dir='output',
        save_ext='jpg',
        gpu=0,
        vgg_weights='models/vgg19_weights_normalized.h5',
        decoder_weights='models/ckp-MST-paper',
        patch_size=3,
        n_clusters_s=3,
        graphPara_smooth=0.1,
        graphPara_max_cycles=3,
        data_format = 'channels_first'):
    assert bool(content) != bool(content_dir), 'Either content or content_dir should be given'
    assert bool(style) != bool(style_dir), 'Either style or style_dir should be given'

    if not os.path.exists(output_dir):
        print('Creating output dir at', output_dir)
        os.makedirs(output_dir)

    # Assume that it is either an h5 file or a name of a TensorFlow checkpoint
    decoder_in_h5 = decoder_weights.endswith('.h5')

    if content:
        content_batch = [content]
    else:
        content_batch = extract_image_names_recursive(content_dir)

    if style:
        style_batch = [style]
    else:
        style_batch = extract_image_names_recursive(style_dir)

    print('Number of content images:', len(content_batch))
    print('Number of style images:', len(style_batch))
    total_output_batch = len(content_batch) * len(style_batch)
    print('Total number of output:', total_output_batch)

    image, content, style, target, encoder, decoder = _build_graph(vgg_weights,
        decoder_weights if decoder_in_h5 else None, alpha, patch_size, data_format=data_format)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    start_time = time.time()
    with tf.Session() as sess:
        if decoder_in_h5:
            sess.run(tf.global_variables_initializer())
        else:
            saver = tf.train.Saver()
            saver.restore(sess, decoder_weights)

        for content_path, style_path in product(content_batch, style_batch):
            content_name = get_filename(content_path)
            content_image = load_image(content_path, content_size, crop)

            style_name = get_filename(style_path)
            style_image = load_image(style_path, style_size, crop)

            style_image = prepare_image(style_image)
            content_image = prepare_image(content_image)
            style_feature = sess.run(encoder, feed_dict={
                image: style_image[np.newaxis,:]
            })
            content_feature = sess.run(encoder, feed_dict={
                image: content_image[np.newaxis,:]
            })

            # style_feature and content_feature information
            Bc,Cc,Hc,Wc = content_feature.shape
            Bs,Cs,Hs,Ws = style_feature.shape
            c_feat_rec_HWxC = np.zeros((Hc*Wc, Cc))

            # reshape content feature
            c_feat_HWxC = BxCxHxW_to_HWxC(content_feature)

            # cluster style feature
            s_feat_HWxC = BxCxHxW_to_HWxC(style_feature)
            s_cluster_centers, s_cluster_labels = cluster_feature(s_feat_HWxC, n_clusters_s)

            # construct D
            graphPara_D = np.double(1 - cosine_similarity(c_feat_HWxC, s_cluster_centers))
            # construct V
            X, Y = np.mgrid[:n_clusters_s, :n_clusters_s]
            graphPara_V = graphPara_smooth*np.float_(np.abs(X-Y))
            # graph cut
            graphPara_sol = fastmin.aexpansion_grid(graphPara_D, graphPara_V, graphPara_max_cycles)
            # ST 
            for label_idx in range(n_clusters_s):
                print("#%d cluster:" % label_idx)
                # select content feature
                c_subset_index = np.argwhere(graphPara_sol == label_idx).flatten()
                c_subset_sample = c_feat_HWxC[c_subset_index,:]
                c_subset_sample = HWxC_to_BxCxHWxW0(c_subset_sample)
                print("c_subset_sample:", c_subset_sample.shape)
                # select cooresponding style feature
                s_subset_index = np.argwhere(s_cluster_labels == label_idx).flatten()
                s_subset_sample = s_feat_HWxC[s_subset_index,:]
                s_subset_sample = HWxC_to_BxCxHWxW0(s_subset_sample)
                print("s_subset_sample:", s_subset_sample.shape)
                # feature transfer
                t_subset_sample = sess.run(target, feed_dict={
                    content: c_subset_sample,
                    style: s_subset_sample
                })
                
                # target feature subset
                t_subset_sample = BxCxHxW_to_HWxC(t_subset_sample)
                c_feat_rec_HWxC[c_subset_index,:] = t_subset_sample
            # reshape to target feature
            target_feature = HWxC_to_BxCxHxW(c_feat_rec_HWxC, Hc, Wc, Cc)
                 
            # obtain output
            output = sess.run(decoder, feed_dict={
                content: content_feature,
                target: target_feature
            })

            filename = '%s_stylized_%s.%s' % (content_name, style_name, save_ext)
            filename = os.path.join(output_dir, filename)
            save_image(filename, output[0], data_format=data_format)
            print('Output image saved at', filename)
        end_time = time.time()
        print('Total outputs=' + str(total_output_batch) + ', total time=' + str(end_time - start_time) + ', average time=' + str((end_time-start_time)/total_output_batch))

def _build_graph(vgg_weights, decoder_weights, alpha, patch_size, data_format):
    if data_format == 'channels_first':
        image = tf.placeholder(shape=(None,3,None,None), dtype=tf.float32)
        content = tf.placeholder(shape=(1,512,None,None), dtype=tf.float32)
        style = tf.placeholder(shape=(1,512,None,None), dtype=tf.float32)
    else:
        image = tf.placeholder(shape=(None,None,None,3), dtype=tf.float32)
        content = tf.placeholder(shape=(1,None,None,512), dtype=tf.float32)
        style = tf.placeholder(shape=(1,None,None,512), dtype=tf.float32)

    target = wct(content, style, num_feature=512)

    weighted_target = target * alpha + (1 - alpha) * content

    with open_weights(vgg_weights) as w:
        vgg = build_vgg(image, w, data_format=data_format)
        encoder = vgg['conv4_1']

    if decoder_weights:
        with open_weights(decoder_weights) as w:
            decoder = build_decoder(weighted_target, w, trainable=False,
                data_format=data_format)
    else:
        decoder = build_decoder(weighted_target, None, trainable=False,
            data_format=data_format)

    return image, content, style, target, encoder, decoder

def BxCxHxW_to_HWxC(feature_BNHW):
    # squeeze: BxNxHxW -> NxHxW
    feature = np.squeeze(feature_BNHW, axis=0)
    # reshape: NxHxW -> NxHW
    C, H, W = feature.shape
    feature = np.reshape(feature, (C, H*W))
    # transpose: NxHW -> HWxN
    feature = np.transpose(feature)
    return feature

def HWxC_to_BxCxHxW(feature_HWxC, H, W, C):
    # transpose: HWxC -> CxHW
    feature = np.transpose(feature_HWxC)
    # reshape: CxHW -> CxHxW
    feature = np.reshape(feature, (C,H,W))
    # expand_dim: CxHxW -> BxCxHxW
    feature = np.expand_dims(feature, axis=0)
    return feature

def HWxC_to_BxCxHWxW0(feature_HWxC):
    # HWxC -> CxHW
    feature = np.transpose(feature_HWxC)
    # CxHW -> BxCxHW
    feature = np.expand_dims(feature, axis=0)
    # BxCxHW -> BxCxHWxW0
    feature = np.expand_dims(feature, axis=3)
    return feature

def cluster_feature(feature_HWxN, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(feature_HWxN)
    cluster_centers = kmeans.cluster_centers_
    cluster_labels  = kmeans.labels_
    return cluster_centers, cluster_labels

if __name__ == '__main__':
    params = get_params(style_transfer)
    parser = argparse.ArgumentParser(description='Multimodal Style Transfer via Graph Cuts')

    parser.add_argument('--content', default=params['content'], help='File path to the content image')
    parser.add_argument('--content_dir', default=params['content_dir'], help="""Directory path to a batch of
        content images""")
    parser.add_argument('--style', default=params['style'], help="""File path to the style image,
        or multiple style images separated by commas if you want to do style
        interpolation or spatial control""")
    parser.add_argument('--style_dir', default=params['style_dir'],  help="""Directory path to a batch of
        style images""")
    parser.add_argument('--vgg_weights', default=params['vgg_weights'],
        help='Path to the weights of the VGG19 network')
    parser.add_argument('--decoder_weights', default=params['decoder_weights'],
        help='Path to the decoder')
    parser.add_argument('--content_size', default=params['content_size'],
        type=int, help="""Maximum size for the content image, keeping
        the original size if set to 0""")
    parser.add_argument('--style_size', default=params['style_size'], type=int,
        help="""Maximum size for the style image, keeping the original
        size if set to 0""")
    parser.add_argument('--crop', action='store_true', help="""If set, center
        crop both content and style image before processing""")
    parser.add_argument('--save_ext', default=params['save_ext'],
        help='The extension name of the output image')
    parser.add_argument('--gpu', default=params['gpu'], type=int,
        help='Zero-indexed ID of the GPU to use; for CPU mode set to -1')
    parser.add_argument('--output_dir', default=params['output_dir'],
        help='Directory to save the output image(s)')
    parser.add_argument('--alpha', default=params['alpha'], type=float,
        help="""The weight that controls the degree of stylization. Should be
        between 0 and 1""")
    parser.add_argument('--patch_size', default=params['patch_size'], type=int,
        help="""Patch size for patch matching""")
    parser.add_argument('--n_clusters_s', default=params['n_clusters_s'], type=int,
        help="""number of cluster center of style""")
    parser.add_argument('--graphPara_smooth', default=params['graphPara_smooth'], type=float,
        help="""smooth factor in graph cut""")
    parser.add_argument('--graphPara_max_cycles', default=params['graphPara_max_cycles'], type=int,
        help="""cycle factor in graph cut""")
    parser.add_argument('--data_format', default=params['data_format'],
        help='data_format: channels_first or channels_last')

    args = parser.parse_args()
    style_transfer(**vars(args))