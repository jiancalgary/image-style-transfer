# coding: utf-8

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf

import load_vgg
import utils


def setup():
    """
    
    """
    utils.safe_mkdir("checkpoints")
    utils.safe_mkdir("outputs")


class StyleTransfer(object):
    def __init__(self, content_img, style_img, img_width, img_height):
    
    
   
        self.content_name = str(content_img.split("/")[-1].split(".")[0])
        self.style_name = str(style_img.split("/")[-1].split(".")[0])
        self.img_width = img_width
        self.img_height = img_height
     
        self.content_img = utils.get_resized_image(content_img, img_width, img_height)
        self.style_img = utils.get_resized_image(style_img, img_width, img_height)
        self.initial_img = utils.generate_noise_image(self.content_img, img_width, img_height)

     
        self.content_layer = "conv4_2"
        self.style_layers = ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"]

        # 定义content loss和style loss的权重
        self.content_w = 0.001
        self.style_w = 1

   
        self.style_layer_w = [0.5, 1.0, 1.5, 3.0, 4.0]

   
        self.gstep = tf.Variable(0, dtype=tf.int32, trainable=False, name="global_step")  # global step
        self.lr = 2.0

        utils.safe_mkdir("outputs/%s_%s" % (self.content_name, self.style_name))

    def create_input(self):
      
        with tf.variable_scope("input"):
            self.input_img = tf.get_variable("in_img", 
                                             shape=([1, self.img_height, self.img_width, 3]),
                                             dtype=tf.float32,
                                             initializer=tf.zeros_initializer())

    def load_vgg(self):
      
        self.vgg = load_vgg.VGG(self.input_img)
        self.vgg.load()
        # mean-center
        self.content_img -= self.vgg.mean_pixels
        self.style_img -= self.vgg.mean_pixels

    def _content_loss(self, P, F):
  
        self.content_loss = tf.reduce_sum(tf.square(F - P)) / (4.0 * P.size)
        
    def _gram_matrix(self, F, N, M):
    
        F = tf.reshape(F, (M, N))

        return tf.matmul(tf.transpose(F), F)

    def _single_style_loss(self, a, g):
     
        N = a.shape[3]
        M = a.shape[1] * a.shape[2]

        # 生成feature map的Gram Matrix
        A = self._gram_matrix(a, N, M)
        G = self._gram_matrix(g, N, M)

        return tf.reduce_sum(tf.square(G - A)) / ((2 * N * M) ** 2)

    def _style_loss(self, A):
     
        # 层数（我们用了conv1_1, conv2_1, conv3_1, conv4_1, conv5_1）
        n_layers = len(A)
        # 计算loss
        E = [self._single_style_loss(A[i], getattr(self.vgg, self.style_layers[i]))
             for i in range(n_layers)]
        # 加权求和
        self.style_loss = sum(self.style_layer_w[i] * E[i] for i in range(n_layers))

    def losses(self):
       
        with tf.variable_scope("losses"):
            # contents loss
            with tf.Session() as sess:
                sess.run(self.input_img.assign(self.content_img))
                gen_img_content = getattr(self.vgg, self.content_layer)
                content_img_content = sess.run(gen_img_content)
            self._content_loss(content_img_content, gen_img_content)

            # style loss
            with tf.Session() as sess:
                sess.run(self.input_img.assign(self.style_img))
                style_layers = sess.run([getattr(self.vgg, layer) for layer in self.style_layers])                              
            self._style_loss(style_layers)

            
            self.total_loss = self.content_w * self.content_loss + self.style_w * self.style_loss

    def optimize(self):
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.total_loss, global_step=self.gstep)

    def create_summary(self):
        with tf.name_scope("summary"):
            tf.summary.scalar("contents loss", self.content_loss)
            tf.summary.scalar("style loss", self.style_loss)
            tf.summary.scalar("total loss", self.total_loss)
            self.summary_op = tf.summary.merge_all()

    def build(self):
        self.create_input()
        self.load_vgg()
        self.losses()
        self.optimize()
        self.create_summary()

    def train(self, epoches=300):
        skip_step = 1
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            writer = tf.summary.FileWriter("graphs/style_transfer", sess.graph)
            
            sess.run(self.input_img.assign(self.initial_img))

            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(os.path.dirname("checkpoints/%s_%s_style_transfer/checkpoint" %
                                                                 (self.content_name, self.style_name)))
            if ckpt and ckpt.model_checkpoint_path:
                print("You have pre-trained model, if you do not want to use this, please delete the existing one.")
                saver.restore(sess, ckpt.model_checkpoint_path)

            initial_step = self.gstep.eval()

            for epoch in range(initial_step, epoches):
              
                if epoch >= 5 and epoch < 20:
                    skip_step = 10
               
                elif epoch >= 20:
                    skip_step = 20
                
                sess.run(self.optimizer)
                if (epoch + 1) % skip_step == 0:
                    gen_image, total_loss, summary = sess.run([self.input_img,
                                                               self.total_loss,
                                                               self.summary_op])
                  
                    gen_image = gen_image + self.vgg.mean_pixels 
                    writer.add_summary(summary, global_step=epoch)

                    print("Step {}\n   Sum: {:5.1f}".format(epoch + 1, np.sum(gen_image)))
                    print("   Loss: {:5.1f}".format(total_loss))

                    filename = "outputs/%s_%s/epoch_%d.png" % (self.content_name, self.style_name, epoch)
                    utils.save_image(filename, gen_image)

                   
                    if (epoch + 1) % 20 == 0:
                        saver.save(sess,
                                   "checkpoints/%s_%s_style_transfer/style_transfer" %
                                   (self.content_name, self.style_name), epoch)

if __name__ == "__main__":
    setup()
  
    content_img = "contents/sky.jpg"
    style_img = "styles/starry_night.jpg"

    img_width = 400
    img_height = 300
    # style transfer
    style_transfer = StyleTransfer(content_img, style_img, img_width, img_height)
    style_transfer.build()
    style_transfer.train(300)
