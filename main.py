import os
import tensorflow as tf
import numpy as np
import scipy.misc
import Model

CONTENT_IMG = './images/Fig2a.jpg'
STYLE_IMG = './images/Fig2b.jpg'
TARGET_MASK = './images/Fig2a_guide.jpg'
STYLE_MASK = './images/Fig2b_guide.jpg'
MASK_TYPE = 'simple' # 'simple' 'test' 'colorful'

HARD_WIDTH = 256

INIT_TYPE = 'noise'  # 'content' 'style'
INIT_NOISE_RATIO = 1.0

MODEL_PATH = 'imagenet-vgg-verydeep-19.mat'
FEATURE_POOLING_TYPE = 'avg'  # 'max'
MASK_DOWNSAMPLE_TYPE = 'simple' # 'all' 'inside' 'mean'

CONTENT_LAYERS = ['relu4_2']
CONTENT_LAYERS_WEIGHTS = [1.]
STYLE_LAYERS = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']
STYLE_LAYERS_WEIGHTS = [1., 1., 1., 1., 1.]

CONTENT_LOSS_NORMALIZATION = 1 # 1 for 1./(N * M); 2 for 1./(2. * N**0.5 * M**0.5)
MASK_NORMALIZATION_TYPE = 'square_sum' # 'sum'

CONTENT_WEIGHT = 1.
STYLE_WEIGHT = 500.
TV_WEIGHT = 1e-4

LEARNING_RATE = 1.
ITERATION = 3000
LOG_ITERATION = 100

OUTPUT_DIR = './output'


'''
    read & write & init
'''
def read_image(path):   # read and preprocess
    img = scipy.misc.imread(path)   
    
    #(resize)
    if HARD_WIDTH:
        img = scipy.misc.imresize(img, float(HARD_WIDTH) / img.shape[1])

    img = img.astype(np.float32)
    img = img[np.newaxis, :, :, :]
    img = img - [123.68, 116.779, 103.939]
    return img

def read_mask(path):   # read mask, (optional cluster to masks,), '1'-based
    if MASK_TYPE == 'test':
        rawmask = scipy.misc.imread(path)

        #(resize)
        if HARD_WIDTH:
            rawmask = scipy.misc.imresize(rawmask, float(HARD_WIDTH) / rawmask.shape[1])

        rawmask = rawmask.astype(np.float32) # rgb  
        single = (rawmask.transpose([2, 0, 1]))[0]
        single = single / 255.
        return np.stack([single, 1.-single])

def write_image(path, img):   # postprocess and write
    img = img + [123.68, 116.779, 103.939]
    img = img[0]
    img = np.clip(img, 0, 255).astype('uint8')
    scipy.misc.imsave(path, img)

def get_init_image(init_type, content_img, init_noise_ratio):
    # why [-20, 20]???
    noise_img = np.random.uniform(-20., 20., content_img.shape).astype(np.float32)
    init_img = init_noise_ratio * noise_img + (1. - init_noise_ratio) * content_img
    return init_img


'''
    compute features & masks 
    build net
'''
def compute_features(vgg_weights, input_img, layers):
    input = tf.placeholder(tf.float32, shape=input_img.shape)
    net = Model.build_image_net(input, vgg_weights, FEATURE_POOLING_TYPE)
    features = {}
    with tf.Session() as sess:
        for layer in layers:
            features[layer] = sess.run(net[layer], feed_dict={input: input_img})
    return features

# transpose & reshape
def compute_layer_masks(masks, layers):
    masks_tf = masks.transpose([1,2,0])
    masks_tf = masks_tf[np.newaxis, :, :, :]

    input = tf.placeholder(tf.float32, shape=masks_tf.shape)
    net = Model.build_mask_net(input, MASK_DOWNSAMPLE_TYPE)
    layer_masks = {}
    with tf.Session() as sess:
        for layer in layers:
            out = sess.run(net[layer], feed_dict={input: masks_tf})
            layer_masks[layer] = out[0].transpose([2,0,1])
    return layer_masks

def build_target_net(vgg_weights, target_shape):
    input = tf.Variable( np.zeros(target_shape).astype('float32') )
    net = Model.build_image_net(input, vgg_weights, FEATURE_POOLING_TYPE)
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

    # TODO: use local variable
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

def gram_matrix(x): # tensor
    _, h, w, d = x.get_shape()
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

def sum_style_loss(target_net, style_features, layers, layers_weights):
    style_loss = 0.
    for layer, weight in zip(layers, layers_weights):
        a = style_features[layer]
        x = target_net[layer]
        style_loss += style_layer_loss(a, x) * weight
    style_loss /= float(sum(layers_weights))
    return style_loss    

'''
    main
'''
def  main():
    
'''
init 
'''   
content_img = read_image(CONTENT_IMG)   # read & preprocess

if MASK_TYPE == 'simple':
    target_masks_origin = np.ones(content_img.shape[0:3]).astype(np.float32) #stack
else:
    target_masks_origin = read_mask(TARGET_MASK) 

style_img = read_image(STYLE_IMG)       # read & preprocess

if MASK_TYPE == 'simple':
    style_masks_origin = np.ones(style_img.shape[0:3]).astype(np.float32) #stack
else:
    style_masks_origin = read_mask(STYLE_MASK)

init_img = get_init_image(INIT_TYPE, content_img, INIT_NOISE_RATIO)

# check shape & number of masks
if content_img.shape[1:3] != target_masks_origin.shape[1:3]:
    print('content and mask have different shape')
if style_img.shape[1:3] != style_masks_origin.shape[1:3]:
    print('style and mask have different shape')
if target_masks_origin.shape[0] != style_masks_origin.shape[0]:
    print('content and style have different masks')

'''
compute features & build net
'''
# prepare model
vgg_weights = Model.prepare_model(MODEL_PATH)

# content features
content_features = compute_features(vgg_weights, content_img, CONTENT_LAYERS)
# style features
style_features = compute_features(vgg_weights, style_img, STYLE_LAYERS)

# masks of layers
target_masks = compute_layer_masks(target_masks_origin, STYLE_LAYERS)
style_masks = compute_layer_masks(style_masks_origin, STYLE_LAYERS)

# build net
target_net = build_target_net(vgg_weights, content_img.shape)


'''
loss 
'''
content_loss = sum_content_loss(target_net, content_features, 
                                CONTENT_LAYERS, CONTENT_LAYERS_WEIGHTS,
                                CONTENT_LOSS_NORMALIZATION)

style_masked_loss = sum_masked_style_loss(target_net, style_features, 
                                          target_masks, style_masks, 
                                          STYLE_LAYERS, STYLE_LAYERS_WEIGHTS, 
                                          MASK_NORMALIZATION_TYPE)
#tv_loss
tv_loss = 0.

total_loss = CONTENT_WEIGHT * content_loss + \
             STYLE_WEIGHT * style_masked_loss + \
             TV_WEIGHT * tv_loss


'''
train 
'''
optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
train_op = optimizer.minimize(total_loss)

if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

init_op = tf.global_variables_initializer() # must!
sess = tf.Session()
sess.run(init_op)
sess.run( target_net['input'].assign(init_img) )
for i in range(ITERATION):
    sess.run(train_op)
    if i % LOG_ITERATION == 0:
        print('Iteration %d: loss = %f' % (i, sess.run(total_loss)))
        result = sess.run(target_net['input'])
        output_path = os.path.join(OUTPUT_DIR, 'result_%s.png' % (str(i).zfill(4)))
        write_image(output_path, result)

'''
out
'''
result = sess.run(target_net['input'])
output_path = os.path.join(OUTPUT_DIR, 'result_final.png')
write_image(output_path, result)


if __name__ == '__main__':
    main()





