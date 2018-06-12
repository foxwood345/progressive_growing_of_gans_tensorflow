import tensorflow as tf
from ops import lrelu, conv2d, fully_connect, upscale, Pixl_Norm, avgpool2d, WScaleLayer, MinibatchstateConcat
from utils import save_images
from utils import CelebA
import numpy as np
import scipy

class PGGAN(object):

    # build model
    def __init__(self, batch_size, max_iters, model_path, read_model_path, data, sample_size, sample_path, log_dir,
                 learn_rate, PG, t):

        self.batch_size = batch_size
        self.max_iters = max_iters
        # 이번 단계의 모델 저장 위치
        self.gan_model_path = model_path
        # 이전 단계의 모델 path(이전 단계의 weight를 불러옴)
        self.read_model_path = read_model_path
        self.data_In = data
        # 512 latent size
        self.sample_size = sample_size
        # 이미지가 저장되는 공간
        self.sample_path = sample_path
        self.log_dir = log_dir
        self.learning_rate = learn_rate
        self.pg = PG
        # transition phase, stablization phase인지 True, False
        self.trans = t
        self.log_vars = []
        self.channel = 3
        # 처음에 pg값은 1이므로 output_size는 4가 나옴
        # pg값이 2일때 4 * 2^1 = 8
        # pg값이 3일때 4 * 2^2 = 16
        # 마지막 pg값이 6일때 4 * 2^5 = 128
        self.output_size = 4 * pow(2, PG - 1)

        # real image의 placeholder (discriminator로 들어가는)
        self.images = tf.placeholder(tf.float32, [batch_size, self.output_size, self.output_size, self.channel])
        # z latent vector의 placeholder
        self.z = tf.placeholder(tf.float32, [self.batch_size, self.sample_size])
        # alpha_trainsiton에 대한 Variable
        self.alpha_tra = tf.Variable(initial_value=0.0, trainable=False,name='alpha_tra')

    def build_model_PGGan(self):
        
        # generator에서 만든 fake image
        self.fake_images = self.generate(self.z, pg=self.pg, t=self.trans, alpha_trans=self.alpha_tra)
        # real image를 받은 discriminator에서 나오는 logit (activation을 거친 값이 아님)
        _, self.D_pro_logits = self.discriminate(self.images, reuse=False, pg = self.pg, t=self.trans, alpha_trans=self.alpha_tra)
        # fake image를 받은 discriminator에서 나오는 logit (activation을 거친 값이 아님) , reuse=True
        _, self.G_pro_logits = self.discriminate(self.fake_images, reuse=True,pg= self.pg, t=self.trans, alpha_trans=self.alpha_tra)

        # the defination of loss for D and G
        # D_loss = d_loss_fake + (-d_loss_real)
        self.D_loss = tf.reduce_mean(self.G_pro_logits) - tf.reduce_mean(self.D_pro_logits)
        self.G_loss = -tf.reduce_mean(self.G_pro_logits)

        # gradient penalty from WGAN-GP
        self.differences = self.fake_images - self.images
        self.alpha = tf.random_uniform(shape=[self.batch_size, 1, 1, 1], minval=0., maxval=1.)
        interpolates = self.images + (self.alpha * self.differences)
        _, discri_logits= self.discriminate(interpolates, reuse=True, pg=self.pg, t=self.trans, alpha_trans=self.alpha_tra)
        gradients = tf.gradients(discri_logits, [interpolates])[0]

        ##2 norm
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2, 3]))
        self.gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
        tf.summary.scalar("gp_loss", self.gradient_penalty)

        # 본래 D_loss
        self.D_origin_loss = self.D_loss

        # WGAN-GP을 적용한 D_loss
        self.D_loss += 10 * self.gradient_penalty
        # 0에서 멀어지지 않게 하기 위한 term을 추가
        self.D_loss += 0.001 * tf.reduce_mean(tf.square(self.D_pro_logits - 0.0))

        # G_loss와 D_loss를 list에 기록함
        self.log_vars.append(("generator_loss", self.G_loss))
        self.log_vars.append(("discriminator_loss", self.D_loss))

        # network에서 trainable한 모든 variable(weight)를 가져옴
        t_vars = tf.trainable_variables()
        # 이 중에서 discriminator의 d_vars
        self.d_vars = [var for var in t_vars if 'dis' in var.name]

        # discriminator의 weight(partmater)와 총 갯수를 출력해줌
        total_para = 0
        for variable in self.d_vars:
            shape = variable.get_shape()
            print (variable.name, shape)
            variable_para = 1
            for dim in shape:
                variable_para *= dim.value
            total_para += variable_para
        print ("The total para of D", total_para)

        self.g_vars = [var for var in t_vars if 'gen' in var.name]

        # generator의 weight(partmater)와 총 갯수를 출력해줌
        total_para2 = 0
        for variable in self.g_vars:
            shape = variable.get_shape()
            print (variable.name, shape)
            variable_para = 1
            for dim in shape:
                variable_para *= dim.value
            total_para2 += variable_para
        print ("The total para of G", total_para2)

        #save the variables , which remain unchanged
        self.d_vars_n = [var for var in self.d_vars if 'dis_n' in var.name]
        self.g_vars_n = [var for var in self.g_vars if 'gen_n' in var.name]

        # remove the new variables for the new model
        # 처음에 pg값은 1이므로 output_size는 4가 나옴
        # pg값이 2일때 4 * 2^1 = 8
        # pg값이 3일때 4 * 2^2 = 16
        # weight 중 새로 추가된 해상도에 대한 weight를 제외한 weight list
        self.d_vars_n_read = [var for var in self.d_vars_n if '{}'.format(self.output_size) not in var.name]
        self.g_vars_n_read = [var for var in self.g_vars_n if '{}'.format(self.output_size) not in var.name]

        # save the rgb variables, which remain unchanged
        # 전체 fromRGB, toRGB
        self.d_vars_n_2 = [var for var in self.d_vars if 'dis_y_rgb_conv' in var.name]
        self.g_vars_n_2 = [var for var in self.g_vars if 'gen_y_rgb_conv' in var.name]

        # 이번 단계에 나온 fromRGB, toRGB가 아닌 기존의 fromRGB, toRGB 
        self.d_vars_n_2_rgb = [var for var in self.d_vars_n_2 if '{}'.format(self.output_size) not in var.name]
        self.g_vars_n_2_rgb = [var for var in self.g_vars_n_2 if '{}'.format(self.output_size) not in var.name]

        print ("d_vars", len(self.d_vars))
        print ("g_vars", len(self.g_vars))

        print ("self.d_vars_n_read", len(self.d_vars_n_read))
        print ("self.g_vars_n_read", len(self.g_vars_n_read))

        print ("d_vars_n_2_rgb", len(self.d_vars_n_2_rgb))
        print ("g_vars_n_2_rgb", len(self.g_vars_n_2_rgb))

        # for n in self.d_vars:
        #     print (n.name)

        self.g_d_w = [var for var in self.d_vars + self.g_vars if 'bias' not in var.name]

        print ("self.g_d_w", len(self.g_d_w))

        # generator와 discriminator의 weight를 저장을 담당하는 Saver를 만듬
        # saver : 전체 모델 weights, r_saver : 이전 단계의 모델 weighs
        self.saver = tf.train.Saver(self.d_vars + self.g_vars)
        self.r_saver = tf.train.Saver(self.d_vars_n_read + self.g_vars_n_read)

        # 만약 toRGB, fromRGB이 있다면 이것도 저장을 담당하는 Saver를 만듬
        if len(self.d_vars_n_2_rgb + self.g_vars_n_2_rgb):
            self.rgb_saver = tf.train.Saver(self.d_vars_n_2_rgb + self.g_vars_n_2_rgb)

        for k, v in self.log_vars:
            tf.summary.scalar(k, v)

    # do train
    def train(self):
        # iteration이 지날때 마다 linearly alpha를 증가시킴
        step_pl = tf.placeholder(tf.float32, shape=None)
        alpha_tra_assign = self.alpha_tra.assign(step_pl / self.max_iters)

        # optimizer를 이용하여 loss를 줄이는 방향으로 variable을 업데이트해나간다.
        opti_D = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.0 , beta2=0.99).minimize(
            self.D_loss, var_list=self.d_vars)
        opti_G = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.0 , beta2=0.99).minimize(
            self.G_loss, var_list=self.g_vars)

        # 변수들을 초기화하는 그래프
        init = tf.global_variables_initializer()
        # GPU 메모리를 필요할때 마다 조금씩 늘리도록 한다. 
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            
            # 변수들을 초기화하는 그래프를 돌림
            sess.run(init)
            
            summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(self.log_dir, sess.graph)

            # progressive 값이 1, 7을 제외한 값이라면 
            if self.pg != 1 and self.pg != 7:
                # transition이 일어나는 때라면
                if self.trans:

                    # d_vars_n_read와 g_vars_n_read를 담당하는 r_saver가 이전 progressive에서
                    # training 하여 저장했던 weight들을 불러온다
                    self.r_saver.restore(sess, self.read_model_path)
                    # 이전에 학습했던 d_vars_n_2_rgb와 g_vars_n_2_rgb를 불러온다
                    self.rgb_saver.restore(sess, self.read_model_path)

                # trainsition이 일어나는 단계가 아니라면
                else:
                    # 전체 이전 단계의 전체 weight를 불러온다.
                    self.saver.restore(sess, self.read_model_path)

            step = 0
            batch_num = 0
            # 16 x 32000 = 512,000개의 실제 이미지를 discriminator에게 보여줄 때까지 돌림
            while step <= self.max_iters:

                # optimization D
                n_critic = 1
                if self.pg == 5 and self.trans:
                    n_critic = 1

                for i in range(n_critic):
                    # 512 짜리 latent vector를 만듬
                    sample_z = np.random.normal(size=[self.batch_size, self.sample_size])
                    # 실제 이미지의 path를 batch단위씩 얻음 (16개씩)
                    train_list = self.data_In.getNextBatch(batch_num, self.batch_size)
                    # 이 path를 현재 단계의 아웃풋 사이즈만큼 리사이즈 시킴
                    # 예를 들어 1단계에서 output_size가 4이므로 4x4로 resize한다.
                    realbatch_array = CelebA.getShapeForData(train_list, resize_w=self.output_size)
                    # 만약 transition(fade)가 일어나는 차례라면
                    if self.trans and self.pg != 0:

                        alpha = np.float(step) / self.max_iters
                        # 이미지 해상도 변경
                        # https://stackoverflow.com/questions/37119071/scipy-rotate-and-zoom-an-image-without-changing-its-dimensions
                        low_realbatch_array = scipy.ndimage.zoom(realbatch_array, zoom=[1,0.5,0.5,1])
                        low_realbatch_array = scipy.ndimage.zoom(low_realbatch_array, zoom=[1, 2, 2, 1])
                        # resolution transition중에 실제이미지의 resolution 사이의 보간을 한다.
                        realbatch_array = alpha * realbatch_array + (1 - alpha) * low_realbatch_array

                    sess.run(opti_D, feed_dict={self.images: realbatch_array, self.z: sample_z})
                    # 다음 번 배치
                    batch_num += 1

                # optimization G
                sess.run(opti_G, feed_dict={self.z: sample_z})

                summary_str = sess.run(summary_op, feed_dict={self.images: realbatch_array, self.z: sample_z})
                summary_writer.add_summary(summary_str, step)
                # the alpha of fake_in process
                # 1씩 증가하는 step을 step_placeholder에 넣고 alpha_transition값을 계산
                sess.run(alpha_tra_assign, feed_dict={step_pl: step})
                # 400번째 step마다 loss 출력하고 이미지를 저장
                if step % 400 == 0:

                    D_loss, G_loss, D_origin_loss, alpha_tra = sess.run([self.D_loss, self.G_loss, self.D_origin_loss,self.alpha_tra], feed_dict={self.images: realbatch_array, self.z: sample_z})
                    print("PG %d, step %d: D loss=%.7f G loss=%.7f, D_or loss=%.7f, opt_alpha_tra=%.7f" % (self.pg, step, D_loss, G_loss, D_origin_loss, alpha_tra))

                    realbatch_array = np.clip(realbatch_array, -1, 1)
                    save_images(realbatch_array[0:self.batch_size], [2, self.batch_size/2],
                                '{}/{:02d}_real.png'.format(self.sample_path, step))

                    # 만약 transition 단계인 경우에
                    if self.trans and self.pg != 0:

                        low_realbatch_array = np.clip(low_realbatch_array, -1, 1)

                        save_images(low_realbatch_array[0:self.batch_size], [2, self.batch_size / 2],
                                    '{}/{:02d}_real_lower.png'.format(self.sample_path, step))
                   
                    fake_image = sess.run(self.fake_images,
                                          feed_dict={self.images: realbatch_array, self.z: sample_z})
                    fake_image = np.clip(fake_image, -1, 1)
                    save_images(fake_image[0:self.batch_size], [2, self.batch_size/2], '{}/{:02d}_train.png'.format(self.sample_path, step))
                
                # 4000번째 마다 g_vars와 d_vars (network의 weight)를 중간저장함
                if np.mod(step, 4000) == 0 and step != 0:
                    self.saver.save(sess, self.gan_model_path)
                step += 1
            # max_iter 끝나고 최종 모델을 저장함 
            save_path = self.saver.save(sess, self.gan_model_path)
            print ("Model saved in file: %s" % save_path)

        tf.reset_default_graph()


    def discriminate(self, conv, reuse=False, pg=1, t=False, alpha_trans=0.01):

        #dis_as_v = []
        with tf.variable_scope("discriminator") as scope:

            if reuse == True:
                scope.reuse_variables()

            # transition이 True라면(즉, 해상도를 두배로 할 시에)
            if t:
                conv_iden = avgpool2d(conv)
                #from RGB
                # pg=2라면 get_nf에서 512가 나옴
                # name은 dis_y_rgb_conv_(현재 처리하는 해상도/2)
                conv_iden = lrelu(conv2d(conv_iden, output_dim= self.get_nf(pg - 2), k_w=1, k_h=1, d_h=1, d_w=1,
                           name='dis_y_rgb_conv_{}'.format(conv_iden.shape[1])))
            # fromRGB
            # pg=1일때에 get_nf(pg-1)에서 512 나옴
            # pg=2일때에 get_nf(pg-1)에서 512 나옴
            # pg=3일때에 get_nf(pg-1)에서 256 나옴
            # name은 dis_y_rgb_conv_(현재 처리하는 해상도)
            conv = lrelu(conv2d(conv, output_dim=self.get_nf(pg - 1), k_w=1, k_h=1, d_w=1, d_h=1, name='dis_y_rgb_conv_{}'.format(conv.shape[1])))
           # pg=1 일때는 for문 돌지 않음
           # pg=2 일때 i=0나옴
           # pg=3 일때 i=0, 1 나옴
            for i in range(pg - 1):
                # 기본 kernel size는 3x3
                # pg=2이고 i=0일때 get_nf(pg-1-i)에서 512가 나옴
                conv = lrelu(conv2d(conv, output_dim=self.get_nf(pg - 1 - i), d_h=1, d_w=1,
                                    name='dis_n_conv_1_{}'.format(conv.shape[1])))
                # pg=2 이고 i=0일때 get_nf(pg-2-i)에서 512가 나나옴 
                conv = lrelu(conv2d(conv, output_dim=self.get_nf(pg - 2 - i), d_h=1, d_w=1,
                                                      name='dis_n_conv_2_{}'.format(conv.shape[1])))
                conv = avgpool2d(conv, 2)
                # 첫번째 block이고 transition이 True라면 Blending시킴
                if i == 0 and t:
                    conv = alpha_trans * conv + (1 - alpha_trans) * conv_iden

            conv = MinibatchstateConcat(conv)
            # 무조건 channel dimension 512
            conv = lrelu(
                conv2d(conv, output_dim=self.get_nf(1), k_w=3, k_h=3, d_h=1, d_w=1, name='dis_n_conv_1_{}'.format(conv.shape[1])))
            # 무조건 channel dimension 512, VALID라서 feature map의  width, height가 줄어듬
            conv = lrelu(
                conv2d(conv, output_dim=self.get_nf(1), k_w=4, k_h=4, d_h=1, d_w=1, padding='VALID', name='dis_n_conv_2_{}'.format(conv.shape[1])))
            # 1차원으로 reshape
            conv = tf.reshape(conv, [self.batch_size, -1])

            #for D
            output = fully_connect(conv, output_size=1, scope='dis_n_fully')

            return tf.nn.sigmoid(output), output

    def generate(self, z_var, pg=1, t=False, alpha_trans=0.0):

        with tf.variable_scope('generator') as scope:
            # latent vector(batch_size, 512)를 (batch_size, 1, 1, 512)로 바꿈
            de = tf.reshape(z_var, [self.batch_size, 1, 1, tf.cast(self.get_nf(1),tf.int32)])
            de = conv2d(de, output_dim= self.get_nf(1), k_h=4, k_w=4, d_w=1, d_h=1, padding='Other', name='gen_n_1_conv')
            de = Pixl_Norm(lrelu(de))
            # [batch size, 4, 4, 512]로 바꿈
            de = tf.reshape(de, [self.batch_size, 4, 4, tf.cast(self.get_nf(1),tf.int32)])
            de = conv2d(de, output_dim=self.get_nf(1), d_w=1, d_h=1, name='gen_n_2_conv')
            de = Pixl_Norm(lrelu(de))

            #pg=2일때 i=0, for문 1번 실행
            #pg=3일때 i=0,1 for문 2번 실행
            for i in range(pg - 1):
                #pg=2이고 i=0, t=True일때 실행
                #pg=2이고 i=0, t=False일때 실행 x
                #pg=3이고 i=0, t=True 실행x
                #pg=3이고 i=1, t=True 실행 
                if i == pg - 2 and t:
                    #To RGB
                    # 논문에선 upscale을 먼저하고 conv2d(toRGB)를 하도록 되어 있는데 텐서플로우 변수 이름의 중복때문에 어쩔수 없이 conv를 먼저 
                    de_iden = conv2d(de, output_dim=3, k_w=1, k_h=1, d_w=1, d_h=1,
                                     name='gen_y_rgb_conv_{}'.format(de.shape[1]))
                    de_iden = upscale(de_iden, 2)

                de = upscale(de, 2)
                # i=0일때 get_nf(0+1) = 512
                # i=1일때 get_ng(1+1) = 256
                de = Pixl_Norm(lrelu(
                    conv2d(de, output_dim=self.get_nf(i + 1), d_w=1, d_h=1, name='gen_n_conv_1_{}'.format(de.shape[1]))))
                de = Pixl_Norm(lrelu(
                    conv2d(de, output_dim=self.get_nf(i + 1), d_w=1, d_h=1, name='gen_n_conv_2_{}'.format(de.shape[1]))))

            #To RGB
            de = conv2d(de, output_dim=3, k_w=1, k_h=1, d_w=1, d_h=1, name='gen_y_rgb_conv_{}'.format(de.shape[1]))

            if pg == 1:
                return de
            # transition이 True일때
            if t:
                de = (1 - alpha_trans) * de_iden + alpha_trans*de

            else:
                de = de

            return de

    # stage=1일 때 512가 리턴됨
    def get_nf(self, stage):
        return min(1024 / (2 **(stage * 1)), 512)

    def get_fp(self, pg):
        return max(512 / (2 **(pg - 1)), 16)

    def sample_z(self, mu, log_var):
        eps = tf.random_normal(shape=tf.shape(mu))
        return mu + tf.exp(log_var / 2) * eps










