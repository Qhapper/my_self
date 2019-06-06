from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import model
from input_data import get_files

# # 获取一张图片
# def get_one_image(train):
#     # 输入参数：train,训练图片的路径
#     # 返回参数：image，从训练图片中随机抽取一张图片
#     n = len(train)
#     ind = np.random.randint(0, n)
#     img_dir = train[ind]  # 随机选择测试的图片
#
#     img = Image.open(img_dir)
#     plt.imshow(img)
#     plt.show()
#     image = np.array(img)
#     return image


# 测试图片
def evaluate_one_image(image_array):
    with tf.Graph().as_default():
        BATCH_SIZE = 1
        N_CLASSES = 2

        image = tf.cast(image_array, tf.float32)
        image = tf.image.per_image_standardization(image)
        image = tf.reshape(image, [1, 64, 64, 3])

        logit = model.inference(image, BATCH_SIZE, N_CLASSES)

        logit = tf.nn.softmax(logit)

        x = tf.placeholder(tf.float32, shape=[64, 64, 3])

        # you need to change the directories to yours.
        logs_train_dir = './data/save/'

        saver = tf.train.Saver()

        with tf.Session() as sess:

            print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(logs_train_dir)
            # if ckpt and ckpt.model_checkpoint_path:
            if ckpt and ckpt.all_model_checkpoint_paths:
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                print('Loading success, global_step is %s' % global_step)
            else:
                print('No checkpoint file found')

            prediction = sess.run(logit, feed_dict={x: image_array})
            max_index = np.argmax(prediction)
            if max_index == 0:
                result = ('发生火灾的可能性为： %.6f' % prediction[:, 0])
            else:
                result = ('这是正常情况的可能性为： %.6f' % prediction[:, 1])
            return result


# ------------------------------------------------------------------------

if __name__ == '__main__':
    path1 = 'fire1.jpg'
    path2 = 'fire2.jpg'
    path3 = 'fire3.jpg'
    path4= 'normal1.jpg'
    path5= 'normal1.jpg'
    img = Image.open('./data/test_data/' + path3)
    plt.imshow(img)
    plt.show()
    imag = img.resize([64, 64])
    image = np.array(imag)
    print(evaluate_one_image(image))
