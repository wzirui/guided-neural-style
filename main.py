import os
import sys
import tensorflow as tf
import numpy as np
import scipy.misc

import Model
import Parser


'''
    read & write & init
'''
def read_image(path, hard_width):   # read and preprocess
    img = scipy.misc.imread(path)      
    # optional resize, imresize() can only resize uint8 ndarray?
    if hard_width:
        img = scipy.misc.imresize(img, float(hard_width) / img.shape[1])
    img = img.astype(np.float32)
    img = img[np.newaxis, :, :, :]
    img = img - [123.68, 116.779, 103.939]
    return img

def read_mask(path, mask_type, hard_width):   # stacked 0./1. mask
    if mask_type == 'single' or 'double':
        rawmask = scipy.misc.imread(path)
        # optinal resize
        if hard_width:
            rawmask = scipy.misc.imresize(rawmask, float(hard_width) / rawmask.shape[1])
        rawmask = rawmask.astype(np.float32)   
        if len(rawmask.shape) == 3: # rgb
            single = (rawmask.transpose([2, 0, 1]))[0]
        elif len(rawmask.shape) == 2: # grey
            single = rawmask
        single = single / 255.
        if mask_type == 'single':
            return np.stack([single])
        else: # double
            return np.stack([single, 1.-single])

def write_image(path, img):   # postprocess and write
    img = img + [123.68, 116.779, 103.939]
    img = img[0]
    img = np.clip(img, 0, 255).astype('uint8')
    scipy.misc.imsave(path, img)

def get_init_image(content_img, init_noise_ratio):
    # why [-20, 20]???
    noise_img = np.random.uniform(-20., 20., content_img.shape).astype(np.float32)
    init_img = init_noise_ratio * noise_img + (1. - init_noise_ratio) * content_img
    return init_img


'''
    compute features & masks 
    build net
'''
def compute_features(vgg_weights, pooling_type, input_img, layers):
    input = tf.placeholder(tf.float32, shape=input_img.shape)
    net = Model.build_image_net(input, vgg_weights, pooling_type)
    features = {}
    with tf.Session() as sess:
        for layer in layers:
            features[layer] = sess.run(net[layer], feed_dict={input: input_img})
    return features

def compute_layer_masks(masks, layers, ds_type):
    masks_tf = masks.transpose([1,2,0]) # [numberOfMasks, h, w] -> [h, w, masks]
    masks_tf = masks_tf[np.newaxis, :, :, :] # -> [1, h, w, masks]

    input = tf.placeholder(tf.float32, shape=masks_tf.shape)
    net = Model.build_mask_net(input, ds_type) # only do pooling, so no intervention between masks
    layer_masks = {}
    with tf.Session() as sess:
        for layer in layers:
            out = sess.run(net[layer], feed_dict={input: masks_tf})
            layer_masks[layer] = out[0].transpose([2,0,1])
    return layer_masks

def build_target_net(vgg_weights, pooling_type, target_shape):
    input = tf.Variable( np.zeros(target_shape).astype('float32') )
    net = Model.build_image_net(input, vgg_weights, pooling_type)
    net['input'] = input
    return net


'''
    loss
'''
def content_layer_loss(p, x, loss_norm):
    _, h, w, d = p.shape
    M = h * w
    N = d
    if loss_norm  == 1:
        K = 1. / (N * M)
    elif loss_norm == 2:
        K = 1. / (2. * N**0.5 * M**0.5)
    loss = K * tf.reduce_sum( tf.pow((x - p), 2) )
    return loss    

def sum_content_loss(target_net, content_features, layers, layers_weights, loss_norm):
    content_loss = 0.
    for layer, weight in zip(layers, layers_weights):
        p = content_features[layer]
        x = target_net[layer]
        content_loss += content_layer_loss(p, x, loss_norm) * weight
    content_loss /= float(sum(layers_weights))
    return content_loss

def masked_gram(x, mx, mask_norm, N):
    R = mx.shape[0]
    M = mx.shape[1] * mx.shape[2]

    # TODO: use local variable?
    mx = mx.reshape([R, M])
    x = tf.reshape(x, [M, N])
    x = tf.transpose(x) # N * M
    masked_gram = []
    for i in range(R):
        mask = mx[i]
        masked_x = x * mask
        if mask_norm == 'square_sum':
            K = 1. / np.sum(mask**2)
        elif mask_norm == 'sum':
            K = 1. / np.sum(mask)
        gram = K * tf.matmul(masked_x, tf.transpose(masked_x))
        masked_gram.append(gram)
    return tf.stack(masked_gram)

def masked_style_layer_loss(a, ma, x, mx, mask_norm):
    N = a.shape[3]
    R = ma.shape[0]
    K = 1. / (4. * N**2 * R)
    A = masked_gram(a, ma, mask_norm, N)
    G = masked_gram(x, mx, mask_norm, N)
    loss = K * tf.reduce_sum( tf.pow((G - A), 2) )
    return loss

def sum_masked_style_loss(target_net, style_features, target_masks, style_masks, layers, layers_weights, mask_norm):
    style_loss = 0.
    for layer, weight in zip(layers, layers_weights):
        a = style_features[layer]
        ma = style_masks[layer]
        x = target_net[layer]
        mx = target_masks[layer]
        style_loss += masked_style_layer_loss(a, ma, x, mx, mask_norm) * weight
    style_loss /= float(sum(layers_weights))
    return style_loss

def gram_matrix(x): 
    _, h, w, d = x.get_shape() # x is a tensor
    M = h.value * w.value
    N = d.value    
    F = tf.reshape(x, (M, N))
    G = tf.matmul(tf.transpose(F), F)
    return (1./M) * G

def style_layer_loss(a, x):
    N = a.shape[3]
    A = gram_matrix(tf.convert_to_tensor(a))
    G = gram_matrix(x)
    loss = (1./(4 * N**2 )) * tf.reduce_sum(tf.pow((G - A), 2))
    return loss

def sum_style_loss(target_net, style_features, layers, layers_weights): # for testing  
    style_loss = 0.
    for layer, weight in zip(layers, layers_weights):
        a = style_features[layer]
        x = target_net[layer]
        style_loss += style_layer_loss(a, x) * weight
    style_loss /= float(sum(layers_weights))
    return style_loss    

def sum_total_variation_loss(input, shape):
    b, h, w, d = shape
    x = input
    tv_y_size = b * (h-1) * w * d
    tv_x_size = b * h * (w-1) * d
    loss_y = tf.nn.l2_loss(x[:,1:,:,:] - x[:,:-1,:,:])  # l2_loss() use 1/2 factor
    loss_y /= tv_y_size
    loss_x = tf.nn.l2_loss(x[:,:,1:,:] - x[:,:,:-1,:]) 
    loss_x /= tv_x_size
    loss = 2 * (loss_y + loss_x)
    loss = tf.cast(loss, tf.float32) # ?
    return loss


'''
    main
'''
def  main(args):

    '''
    init 
    '''  
    # read images
    content_img = read_image(args.content_img, args.hard_width)   # read & preprocess
    style_img = read_image(args.style_img, args.hard_width)       # read & preprocess

    # get stacked 0./1. masks
    if args.mask_type == 'simple':
        target_masks_origin = np.ones(content_img.shape[0:3]).astype(np.float32)
        style_masks_origin = np.ones(style_img.shape[0:3]).astype(np.float32)
    else:
        target_masks_origin = read_mask(args.target_mask, args.mask_type, args.hard_width) 
        style_masks_origin = read_mask(args.style_mask, args.mask_type, args.hard_width)

    # init img
    init_img = get_init_image(content_img, args.init_noise_ratio)

    # check shape & number of masks
    if content_img.shape[1:3] != target_masks_origin.shape[1:3]:
        print('content and mask have different shape')
        sys.exit(0)
    if style_img.shape[1:3] != style_masks_origin.shape[1:3]:
        print('style and mask have different shape')
        sys.exit(0)
    if target_masks_origin.shape[0] != style_masks_origin.shape[0]:
        print('content and style have different masks')
        sys.exit(0)

    '''
    compute features & build net
    '''
    # prepare model weights
    vgg_weights = Model.prepare_model(args.model_path)

    # feature maps of specific layers
    content_features = compute_features(vgg_weights, args.feature_pooling_type, 
        content_img, args.content_layers)   
    style_features = compute_features(vgg_weights, args.feature_pooling_type, 
        style_img, args.style_layers)

    # masks of specific layers
    target_masks = compute_layer_masks(target_masks_origin, args.style_layers, 
        args.mask_downsample_type)
    style_masks = compute_layer_masks(style_masks_origin, args.style_layers, 
        args.mask_downsample_type)

    # build net
    target_net = build_target_net(vgg_weights, args.feature_pooling_type, content_img.shape)


    '''
    loss 
    '''
    content_loss = sum_content_loss(target_net, content_features, 
                                    args.content_layers, args.content_layers_weights,
                                    args.content_loss_normalization)

    style_masked_loss = sum_masked_style_loss(target_net, style_features, 
                                              target_masks, style_masks, 
                                              args.style_layers, args.style_layers_weights, 
                                              args.mask_normalization_type)

    tv_loss = sum_total_variation_loss(target_net['input'], content_img.shape)

    total_loss = args.content_weight * content_loss + \
                 args.style_weight * style_masked_loss + \
                 args.tv_weight * tv_loss


    '''
    train 
    '''
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    if args.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(args.learning_rate)
        train_op = optimizer.minimize(total_loss)
        #init
        init_op = tf.global_variables_initializer() # must! Adam has some varibales to init
        sess = tf.Session()
        sess.run(init_op)
        sess.run( target_net['input'].assign(init_img) )
        #train
        for i in range(args.iteration):
            sess.run(train_op)
            if i % args.log_iteration == 0:
                print('Iteration %d: loss = %f' % (i+1, sess.run(total_loss)))
                result = sess.run(target_net['input'])
                output_path = os.path.join(args.output_dir, 'result_%s.png' % (str(i).zfill(4)))
                write_image(output_path, result)
    
    elif args.optimizer == 'lbfgs':
        optimizer = tf.contrib.opt.ScipyOptimizerInterface(
            total_loss, method='L-BFGS-B',
            options={'maxiter': args.iteration,
                     'disp': args.log_iteration})   
        # init  
        init_op = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init_op)
        sess.run( target_net['input'].assign(init_img) )
        # train
        optimizer.minimize(sess)    


    '''
    out
    '''
    print('Iteration %d: loss = %f' % (args.iteration, sess.run(total_loss)))
    result = sess.run(target_net['input'])
    output_path = os.path.join(args.output_dir, 'result_final.png')
    write_image(output_path, result)


if __name__ == '__main__':   
    args = Parser.parse_args()
    main(args)





