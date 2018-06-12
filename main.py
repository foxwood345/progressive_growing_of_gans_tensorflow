import tensorflow as tf

from utils import CelebA
from utils import mkdir_p
from PGGAN import PGGAN

# argument역할을 하는 flag 설정
flags = tf.app.flags

flags.DEFINE_integer("OPER_FLAG", 0, "the flag of opertion: 0 is for training ")
flags.DEFINE_string("path" , '../../data/img_align_celeba/', "the path of training data, for example /home/hehe/celebA/")
flags.DEFINE_integer("batch_size", 16, "batch size")
flags.DEFINE_integer("max_iters", 32000, "the maxmization of training number")
flags.DEFINE_float("learn_rate", 0.0001, "the learning rate for G and D networks")
flags.DEFINE_float("flag", 11, "the FLAG of gan training process")

FLAGS = flags.FLAGS
if __name__ == "__main__":

    root_log_dir = "./PGGanCeleba/logs/celeba_test2"
    mkdir_p(root_log_dir)
    batch_size = FLAGS.batch_size
    max_iters = FLAGS.max_iters
    # latent vector size
    sample_size = 512
    GAN_learn_rate = FLAGS.learn_rate

    OPER_FLAG = FLAGS.OPER_FLAG
    # celebA 데이터 loading을 담당하는 클래스
    data_In = CelebA(FLAGS.path)
    print ("the num of dataset", len(data_In.image_list))
    # training 모드일 경우
    if OPER_FLAG == 0:
        # progressive를 나타내는 단계
        # 1에선 trainsition(fade)이 일어나지 않는 첫번째 단계
        # 첫번째 2는 trainsition(fade)이 일어남. 두번째 2는 transition(fade)이 일어나지 않음.
        fl = [1,2,2,3,3,4,4,5,5, 6, 6]
        # r_fl는 이전 단계의 저장했던 네트워크 weights를 불러오는 순서
        # 예를 들어 첫번째 2단계(fl=2)에선 1단계의 weights를 불러와야한다.
        r_fl = [1,1,2,2,3,3,4,4,5, 5, 6]

        # 0 부터 10까지 for문 총 11번 돌림 (f1의 길이만큼) 1, 2, 2, 3,3 ,... 6,6
        for i in range(FLAGS.flag):
            
            # t는 transition를 조절하는 값이다.
            # 짝수 번째일때에는 trans를 하지 않고 홀수번째에 trans를 한다.
            t = False if (i % 2 == 0) else True
            # 이번 단계에 weight를 write하는 path 설정
            pggan_checkpoint_dir_write = "./model_pggan_{}/{}/".format(OPER_FLAG, fl[i])
            # 이미지를 저장할 path
            # 첫번째의 경우 fl[0] == 1, t == Flase,  4x4
            # 두번째의 경우 fl[1] == 2, t == True, 4x4에서 8x8로 뛰기 때문에 transition 일어남
            sample_path = "./PGGanCeleba/{}/sample_{}_{}".format(FLAGS.OPER_FLAG, fl[i], t)
            mkdir_p(pggan_checkpoint_dir_write)
            mkdir_p(sample_path)
            # 이전 단계에 저장해두었던 네트워크의 weight들
            # 예를 들어 현재 pg=2이고 transition=True이면 pg=1에서 저장해둔 weight의 path
            pggan_checkpoint_dir_read = "./model_pggan_{}/{}/".format(OPER_FLAG, r_fl[i])

            # loop를 돌면서 달라지는 값은 model checkpoint 주소와 PG와 t값이다.
            # PGGAN 객체를 만듬
            pggan = PGGAN(batch_size=batch_size, max_iters=max_iters,
                            model_path=pggan_checkpoint_dir_write, read_model_path=pggan_checkpoint_dir_read,
                            data=data_In, sample_size=sample_size,
                            sample_path=sample_path, log_dir=root_log_dir, learn_rate=GAN_learn_rate, PG= fl[i],
                            t=t)
            # model을 build
            pggan.build_model_PGGan()
            # train
            pggan.train()











