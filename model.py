"""
CycleGAN模型
"""

from modules import generator, discriminator
from layers import *
import tensorflow as tf
import time
from utils import image_generator, visual_grid, DataPool
from scipy.misc import imsave, imread, imresize
import os
import glob


class GANModel:
    """
    GAN模型
    """

    def _create_placeholders(self):
        """
        创建占位符
        :return:
        """
        self.realA_ph = tf.placeholder(
            tf.float32,
            (None, self.args.imsize, self.args.imsize, 3),
            name='real_a'
        )
        self.realB_ph = tf.placeholder(
            tf.float32,
            (None, self.args.imsize, self.args.imsize, 3),
            name='real_b'
        )
        self.fakeA_ph = tf.placeholder(
            tf.float32,
            (None, self.args.imsize, self.args.imsize, 3),
            name='fake_a_sample'
        )
        self.fakeB_ph = tf.placeholder(
            tf.float32,
            (None, self.args.imsize, self.args.imsize, 3),
            name='fake_b_sample'
        )

    def _create_GANs(self):
        """
        创建生成器、判别器相应符号
        :return:
        """
        self.fakeA = generator(self.realB_ph, False, name='Gb2a')
        self.fakeB = generator(self.realA_ph, False, name='Ga2b')
        self.cycA = generator(self.fakeB, True, name='Gb2a')
        self.cycB = generator(self.fakeA, True, name='Ga2b')
        self.Da_fake = discriminator(self.fakeA, False, name='Da')
        self.Db_fake = discriminator(self.fakeB, False, name='Db')
        self.Da_real = discriminator(self.realA_ph, True, name='Da')
        self.Db_real = discriminator(self.realB_ph, True, name='Db')
        self.Da_fake_sample = discriminator(self.fakeA_ph, True, name='Da')
        self.Db_fake_sample = discriminator(self.fakeB_ph, True, name='Db')

    def _create_losses(self):
        """
        创建损失函数符号
        :return:
        """
        self.lossCycA = abs_criterion(self.realA_ph, self.cycA, name='Loss_Cyc_A')
        self.lossCycB = abs_criterion(self.realB_ph, self.cycB, name='Loss_Cyc_B')
        with tf.variable_scope('Loss_A2B'):
            self.lossGa2b = tf.add(
                mae_criterion(self.Db_fake, tf.ones_like(self.Db_fake, name='soft_ones_a2b'), name='Loss_Ga2b'),
                tf.add(
                    self.args.clambda * self.lossCycA,
                    self.args.clambda * self.lossCycB,
                    name='Cycle_Loss'
                ),
                name='Total_Loss'
            )
        with tf.variable_scope('Loss_B2A'):
            self.lossGb2a = tf.add(
                mae_criterion(self.Da_fake, tf.ones_like(self.Da_fake, name='soft_ones_b2a'), name='Loss_Gb2a'),
                tf.add(
                    self.args.clambda * self.lossCycA,
                    self.args.clambda * self.lossCycB,
                    name='Cycle_Loss'
                ),
                name='Total_Loss'
            )

        self.lossDa_real = mae_criterion(self.Da_real, tf.ones_like(self.Da_real, name='soft_ones_A'),
                                         name='Loss_Da_real')
        self.lossDb_real = mae_criterion(self.Db_real, tf.ones_like(self.Db_real, name='soft_ones_B'),
                                         name='Loss_Db_real')
        self.lossDa_fake = mae_criterion(self.Da_fake_sample, tf.zeros_like(self.Da_fake_sample,
                                                                            name='soft_zeros_A'), name='Loss_Da_fake')
        self.lossDb_fake = mae_criterion(self.Db_fake_sample, tf.zeros_like(self.Db_fake_sample,
                                                                            name='soft_zeros_B'), name='loss_Db_fake')
        with tf.variable_scope('Loss_Da'):
            self.lossDa = (self.lossDa_real + self.lossDa_fake) * 0.5
        with tf.variable_scope('Loss_Db'):
            self.lossDb = (self.lossDb_real + self.lossDb_fake) * 0.5

    def _collect_vars(self):
        """
        搜集可训练变量
        :return:
        """
        t_vars = tf.trainable_variables()
        self.db_vars = [var for var in t_vars if 'Db' in var.name]
        self.da_vars = [var for var in t_vars if 'Da' in var.name]
        self.g_vars_a2b = [var for var in t_vars if 'Ga2b' in var.name]
        self.g_vars_b2a = [var for var in t_vars if 'Gb2a' in var.name]

    def _create_summaries(self):
        """
        创建TensorBoard可记录量
        :return:
        """
        self.saver = tf.train.Saver()

        self.g_a2b_sum = tf.summary.scalar("g_loss_a2b", self.lossGa2b)
        self.g_b2a_sum = tf.summary.scalar("g_loss_b2a", self.lossGb2a)
        self.db_loss_sum = tf.summary.scalar("db_loss", self.lossDb)
        self.da_loss_sum = tf.summary.scalar("da_loss", self.lossDa)
        self.db_loss_real_sum = tf.summary.scalar("db_loss_real", self.lossDb_real)
        self.db_loss_fake_sum = tf.summary.scalar("db_loss_fake", self.lossDb_fake)
        self.da_loss_real_sum = tf.summary.scalar("da_loss_real", self.lossDa_real)
        self.da_loss_fake_sum = tf.summary.scalar("da_loss_fake", self.lossDa_fake)
        self.db_sum = tf.summary.merge(
            [self.db_loss_sum, self.db_loss_real_sum, self.db_loss_fake_sum]
        )
        self.da_sum = tf.summary.merge(
            [self.da_loss_sum, self.da_loss_real_sum, self.da_loss_fake_sum]
        )
        self.p_img = tf.placeholder(tf.float32, shape=[1, self.args.imsize * 6, self.args.imsize * 4, 3])
        self.img_op = tf.summary.image('sample', self.p_img)

    def _create_opts(self):
        """
        创建优化目标
        :return:
        """
        self.da_optim = tf.train.RMSPropOptimizer(self.args.lr_d) \
            .minimize(self.lossDa, var_list=self.da_vars)
        self.db_optim = tf.train.RMSPropOptimizer(self.args.lr_d) \
            .minimize(self.lossDb, var_list=self.db_vars)
        self.g_a2b_optim = tf.train.RMSPropOptimizer(self.args.lr_g) \
            .minimize(self.lossGa2b, var_list=self.g_vars_a2b)
        self.g_b2a_optim = tf.train.RMSPropOptimizer(self.args.lr_g) \
            .minimize(self.lossGb2a, var_list=self.g_vars_b2a)

    def __init__(self, args):
        """
        创建整个模型
        :param args:
        """
        self.sess = tf.Session()
        self.args = args
        self._create_placeholders()
        self._create_GANs()
        self._create_losses()
        self._create_summaries()
        self._collect_vars()

    def train(self):
        """
        训练
        :return:
        """
        self._create_opts()

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        self.writer = tf.summary.FileWriter(self.args.logdir, self.sess.graph)

        start_time = time.time()

        if self.load(self.args.checkpointdir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        realAtrain = image_generator(os.path.join(self.args.datadir, 'trainA'), 1,
                                     resize=(self.args.imsize, self.args.imsize), value_mode='sigmoid')
        realBtrain = image_generator(os.path.join(self.args.datadir, 'trainB'), 1,
                                     resize=(self.args.imsize, self.args.imsize), value_mode='sigmoid')

        fakeApool = DataPool(self.args.pool_size)
        fakeBpool = DataPool(self.args.pool_size)
        realApool = DataPool(self.args.pool_size)
        realBpool = DataPool(self.args.pool_size)

        counter = 0

        for epoch in range(self.args.nb_epoch):

            for idx in range(0, self.args.nb_batch):

                realAimage = next(realAtrain)
                realBimage = next(realBtrain)

                # Forward G network
                fake_A, fake_B = self.sess.run([self.fakeA, self.fakeB],
                                               feed_dict={
                                                   self.realA_ph: realAimage,
                                                   self.realB_ph: realBimage
                                               })
                realApool.push(realAimage)
                realBpool.push(realBimage)
                fakeApool.push(fake_A)
                fakeBpool.push(fake_B)

                # Update D network
                _, summary_str = self.sess.run([self.db_optim, self.db_sum],
                                               feed_dict={
                                                   self.realB_ph: realBpool.all(),
                                                   self.fakeB_ph: fakeBpool.all()
                                               })
                self.writer.add_summary(summary_str, counter)
                # Update D network
                _, summary_str = self.sess.run([self.da_optim, self.da_sum],
                                               feed_dict={
                                                   self.realA_ph: realApool.all(),
                                                   self.fakeA_ph: fakeApool.all()
                                               })
                self.writer.add_summary(summary_str, counter)
                # Update G network
                _, summary_str = self.sess.run([self.g_a2b_optim, self.g_a2b_sum],
                                               feed_dict={
                                                   self.realA_ph: realAimage,
                                                   self.realB_ph: realBimage
                                               })
                self.writer.add_summary(summary_str, counter)
                # Update G network
                _, summary_str = self.sess.run([self.g_b2a_optim, self.g_b2a_sum],
                                               feed_dict={
                                                   self.realA_ph: realAimage,
                                                   self.realB_ph: realBimage
                                               })
                self.writer.add_summary(summary_str, counter)

                print(
                    ("Epoch: [%2d] [%4d/%4d] time: %4.4f" % (epoch, idx, self.args.nb_batch, time.time() - start_time)))

                counter += 1

                if counter % self.args.sample_freq == 1:
                    self.sample_model(self.args.sampledir, epoch, idx, counter)

                if counter % 1000 == 2:
                    self.save(self.args.checkpointdir, counter)

    def sample_model(self, path, epoch, idx, count):
        """
        保存样例
        :param path: 路径
        :param epoch: epoch数
        :param idx: batch数
        :param count: 总计数
        :return:
        """
        realAtest = image_generator(os.path.join(self.args.datadir, 'testA'), 4,
                                    resize=(self.args.imsize, self.args.imsize), value_mode='sigmoid')
        realBtest = image_generator(os.path.join(self.args.datadir, 'testB'), 4,
                                    resize=(self.args.imsize, self.args.imsize), value_mode='sigmoid')
        real_A = next(realAtest)
        real_B = next(realBtest)
        fake_A, fake_B, cyc_A, cyc_B = self.sess.run(
            [self.fakeA, self.fakeB, self.cycA, self.cycB],
            feed_dict={
                self.realA_ph: real_A,
                self.realB_ph: real_B
            }
        )
        img = visual_grid(
            np.concatenate(
                [real_A, fake_B, cyc_A, real_B, fake_A, cyc_B],
                axis=0
            ),
            (6, 4),
        )

        if self.args.sample_to_file:
            imsave(os.path.join(path, '%4d-%4d.png' % (epoch, idx)), img, 'png')

        img = np.array([img])

        s_img = self.sess.run(self.img_op, feed_dict={self.p_img: img})
        self.writer.add_summary(s_img, count)

    def save(self, path, count):
        """
        保存checkpoint
        :param path: 路径
        :param count: 计数
        :return:
        """
        if not os.path.exists(path):
            os.makedirs(path)

        self.saver.save(self.sess,
                        path,
                        global_step=count)

    def load(self, path):
        """
        加载checkpoint
        :param path: checkpoint路径
        :return: 是否成功
        """
        print(" [*] Reading checkpoint...")

        ckpt = tf.train.get_checkpoint_state(path)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, path)
            return True
        else:
            return False

    def test(self, direction):
        """
        测试
        :param direction: A2B、B2A
        :return:
        """
        savepath = self.args.sampledir
        for file in glob.glob(os.path.join(self.args.datadir, '*.jpg')):
            filename = os.path.split(file)[-1]
            data = imread(file)
            data = imresize(data, (self.args.imsize, self.args.imsize))
            data = np.array([data])
            if direction == 'A2B':
                ret = self.sess.run(
                    self.fakeB,
                    {
                        self.realA_ph: data
                    }
                )
            elif direction == 'B2A':
                ret = self.sess.run(
                    self.fakeA,
                    {
                        self.realB_ph: data
                    }
                )
            imsave(os.path.join(savepath, filename), ret[0], 'png')
