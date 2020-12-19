## library
import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import *
from dataset import *
from util import *

import itertools
import matplotlib.pyplot as plt

from torchvision import transforms

MEAN = 0.5
STD = 0.5

NUM_WORKER = 0

'''
def G_loss(output=None): # loss function for encription network G
    loss = torch.mean(torch.log(1-output))
    return loss
'''

def train(args):
    ## training parmeters
    mode = args.mode
    train_continue = args.train_continue

    lr = args.lr
    batch_size = args.batch_size
    num_epoch = args.num_epoch

    data_dir = args.data_dir
    ckpt_dir = args.ckpt_dir
    log_dir = args.log_dir
    result_dir = args.result_dir

    task = args.task
    opts = [args.opts[0], np.asarray(args.opts[1:]).astype(np.float)]

    ny = args.ny
    nx = args.nx
    nch = args.nch
    nker = args.nker

    norm = args.norm

    network = args.network
    learning_type = args.learning_type

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("mode: %s" % mode)
    print("norm: %s" % norm)

    print("learning rate: %.4e" % lr)
    print("batch size: %d" % batch_size)
    print("number of epoch: %d" % num_epoch)

    print("task: %s" % task)
    print("opts: %s" % opts)

    print("network: %s" % network)
    print("learning type: %s" % learning_type)

    print("data dir: %s" % data_dir)
    print("ckpt dir: %s" % ckpt_dir)
    print("log dir: %s" % log_dir)
    print("result dir: %s" % result_dir)

    print("device: %s" % device)

    ## make result directory
    result_dir_train = os.path.join(result_dir, 'train')

    if not os.path.exists(result_dir_train):
        os.makedirs(os.path.join(result_dir_train, 'png', 'a2b'))
        os.makedirs(os.path.join(result_dir_train, 'png', 'b2a'))

    ## train or Test
    if mode == 'train':
        transform_train = transforms.Compose([Resize(shape=(286, 286, nch)),
                                              RandomCrop((ny, nx)),
                                              Normalization(mean=MEAN, std=STD)])

        dataset_train = Dataset(data_dir=os.path.join(data_dir, 'train'),
                                transform=transform_train,
                                task=task, data_type='both')
        loader_train = DataLoader(dataset_train, batch_size=batch_size,
                                  shuffle=True, num_workers=NUM_WORKER)

        # additional variables 
        num_data_train = len(dataset_train)
        num_batch_train = np.ceil(num_data_train / batch_size)

    ## set network

    if network == "DeepEDN":
        netG_a2b = CycleGAN(in_channels=nch, out_channels=nch, nker=nker, norm=norm, nblk=18).to(device) #encription (residal block#=18)
        netG_b2a = CycleGAN(in_channels=nch, out_channels=nch, nker=nker, norm=norm, nblk=18).to(device) #decription (residal block#=18)

        netD_a = Discriminator(in_channels=nch, out_channels=1, nker=nker, norm=norm).to(device) #discriminator for encription
        netROI = SimpleROI(in_channels=nch, out_channels=2048, nker=2*nker, norm=norm, nblk=4).to(device)

        init_weights(netG_a2b, init_type='normal', init_gain=0.02) #from cycle gan(~N(0,0,02))
        init_weights(netG_b2a, init_type='normal', init_gain=0.02)

        init_weights(netD_a, init_type='normal', init_gain=0.02)
        init_weights(netROI, init_type='normal', init_gain=0.02)

    ## loss function definition
    fn_rec = nn.L1Loss().to(device) # reconstrucion loss : dislike cycle loss of cyclegan, we only need L1(F(G(x))-X)
    fn_dis = nn.BCELoss().to(device) # discriminator loss: same as gan loss of cyclegan -binary cross
    fn_enc = nn.BCELoss().to(device) # encription loss: log(1-D(G(x)))
    fn_roi = nn.MSELoss(reduction='mean').to(device) #ROI loss: Mean Square Error(generated cipher image - corresponding medical image)

    #full objective function: fn_rec + fn_dis + fn_enc 

    ## Optimizer
    optimG = torch.optim.Adam(itertools.chain(netG_a2b.parameters(), netG_b2a.parameters()), lr=lr, betas=(0.5, 0.999)) #encription & decription
    optimD = torch.optim.Adam(netD_a.parameters(), lr=lr, betas=(0.5, 0.999)) #Discriminator for A
    optimR = torch.optim.Adam(netROI.parameters(), lr=lr, betas=(0.5, 0.999)) #ROI

    ## set additional function
    fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1) #tensor to numpy
    fn_denorm = lambda x: (x * STD) + MEAN #denormalization

    cmap = None

    ## SummaryWriter for Tensorboard
    writer_train = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))

    ## Learning Network
    st_epoch = 0

    # TRAIN MODE
    if mode == 'train':
        if train_continue == "on":
            netG_a2b, netG_b2a, \
            netD_a, netROI,\
            optimG, optimD, optimR, st_epoch = load(ckpt_dir=ckpt_dir,
                                            netG_a2b=netG_a2b, netG_b2a=netG_b2a, netROI = netROI,
                                            netD_a=netD_a, optimG=optimG, optimD=optimD, optimR=optimR)

        for epoch in range(st_epoch + 1, num_epoch + 1):
            netG_a2b.train()
            netG_b2a.train()
            netD_a.train()
            netROI.train()

            loss_G_a2b_train = []  #genrator a2b (encription)
            loss_G_b2a_train = []  #generator b2a (decription)
            loss_D_a_train = [] #discriminator for A
            loss_ROI_train = [] #loss for ROI network


            for batch, data in enumerate(loader_train, 1):
                input_a = data['data_a'].to(device)
                input_b = data['data_b'].to(device)

                # forward netG
                output_b = netG_a2b(input_a) #encripted image from A to B
                output_a = netG_b2a(input_b) #descripted cypertext from B to A

                recon_b = netG_a2b(output_a) #B->A->B
                recon_a = netG_b2a(output_b) #A->B->A
                output = netROI(output_b) #ROI: generated cipher image will be an input

                # backward netD
                set_requires_grad([netD_a], True)
                optimD.zero_grad()

                # backward netD_a
                pred_real_a = netD_a(input_a) #inputA -> recongize as real
                pred_fake_a = netD_a(output_a.detach()) #G(A) -> recongize as fake

                loss_D_a_real = fn_dis(pred_real_a, torch.ones_like(pred_real_a)) #real_prediction -> 1
                loss_D_a_fake = fn_dis(pred_fake_a, torch.zeros_like(pred_fake_a)) #fake_prediction ->0
                loss_D_a = 0.5 * (loss_D_a_real + loss_D_a_fake)

                loss_D = loss_D_a
                loss_D.backward()
                optimD.step()

                # backward netG

                set_requires_grad([netD_a], False)
                optimG.zero_grad()

                pred_fake_a = netD_a(output_a) #output_a = G(b), a:originalCT, b:encripted C

                loss_G_a2b = fn_enc(pred_fake_a, torch.ones_like(pred_fake_a)) # encription loss: log(1-D(G(x)))
                loss_G_b2a = fn_rec(input_a, recon_a) # reconstrucion loss : dislike cycle loss of cyclegan, we only need L1(F(G(x))-X)


                loss_G = loss_G_a2b + loss_G_b2a + loss_D
                         

                loss_G.backward()
                optimG.step()

                #backward netROI
                output_b = netG_a2b(input_a)
                output_ROI = netROI(output_b)

                loss_ROI = fn_roi(output_ROI, input_b)
                loss_ROI.backward()

                optimR.step()

                # calculate loss function

                loss_G_a2b_train += [loss_G_a2b.item()]
                loss_G_b2a_train += [loss_G_b2a.item()]
                loss_D_train +=[loss_D.item()]
                loss_ROI_train +=[loss_ROI.item()]


                print("TRAIN: EPOCH %04d / %04d | BATCH %04d / %04d | "
                      "GEN a2b %.4f b2a %.4f | "
                      "DISC a %.4f | "
                      "ROI %.4f | " %
                      (epoch, num_epoch, batch, num_batch_train,
                       np.mean(loss_G_a2b_train), np.mean(loss_G_b2a_train),
                       np.mean(loss_D_train), np.mean(loss_ROI_train)))

                if batch % 20 == 0:
                    # save on Tensorboard
                    input_a = fn_tonumpy(fn_denorm(input_a)).squeeze()
                    input_b = fn_tonumpy(fn_denorm(input_b)).squeeze()
                    output_a = fn_tonumpy(fn_denorm(output_a)).squeeze()
                    output_b = fn_tonumpy(fn_denorm(output_b)).squeeze()

                    input_a = np.clip(input_a, a_min=0, a_max=1)
                    input_b = np.clip(input_b, a_min=0, a_max=1)
                    output_a = np.clip(output_a, a_min=0, a_max=1)
                    output_b = np.clip(output_b, a_min=0, a_max=1)

                    id = num_batch_train * (epoch - 1) + batch

                    plt.imsave(os.path.join(result_dir_train, 'png', 'a2b', '%04d_input_a.png' % id), input_a[0],
                               cmap=cmap)
                    plt.imsave(os.path.join(result_dir_train, 'png', 'a2b', '%04d_output_b.png' % id), output_b[0],
                               cmap=cmap)

                    plt.imsave(os.path.join(result_dir_train, 'png', 'b2a', '%04d_input_b.png' % id), input_b[0],
                               cmap=cmap)
                    plt.imsave(os.path.join(result_dir_train, 'png', 'b2a', '%04d_output_a.png' % id), output_a[0],
                               cmap=cmap)

                    writer_train.add_image('input_a', input_a, id, dataformats='NHWC')
                    writer_train.add_image('input_b', input_b, id, dataformats='NHWC')
                    writer_train.add_image('output_a', output_a, id, dataformats='NHWC')
                    writer_train.add_image('output_b', output_b, id, dataformats='NHWC')

            writer_train.add_scalar('loss_G_a2b', np.mean(loss_G_a2b_train), epoch)
            writer_train.add_scalar('loss_G_b2a', np.mean(loss_G_b2a_train), epoch)

            writer_train.add_scalar('loss_D_a', np.mean(loss_D_a_train), epoch)

            writer_train.add_scalar('loss_ROI', np.mean(loss_ROI_train), epoch)

            if epoch % 2 == 0 or epoch == num_epoch:
                save(ckpt_dir=ckpt_dir, epoch=epoch,
                     netG_a2b=netG_a2b, netG_b2a=netG_b2a,
                     netD_a=netD_a, netROI = netROI,
                     optimG=optimG, optimD=optimD, optimR = optimR)

        writer_train.close()


def test(args):
    ## 트레이닝 파라메터 설정하기
    mode = args.mode
    train_continue = args.train_continue

    lr = args.lr
    batch_size = args.batch_size
    num_epoch = args.num_epoch

    data_dir = args.data_dir
    ckpt_dir = args.ckpt_dir
    log_dir = args.log_dir
    result_dir = args.result_dir

    task = args.task
    opts = [args.opts[0], np.asarray(args.opts[1:]).astype(np.float)]

    ny = args.ny
    nx = args.nx
    nch = args.nch
    nker = args.nker

    norm = args.norm

    network = args.network
    learning_type = args.learning_type

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("mode: %s" % mode)

    print("learning rate: %.4e" % lr)
    print("batch size: %d" % batch_size)
    print("number of epoch: %d" % num_epoch)

    print("task: %s" % task)
    print("opts: %s" % opts)

    print("network: %s" % network)
    print("learning type: %s" % learning_type)

    print("data dir: %s" % data_dir)
    print("ckpt dir: %s" % ckpt_dir)
    print("log dir: %s" % log_dir)
    print("result dir: %s" % result_dir)

    print("device: %s" % device)

    ## 디렉토리 생성하기
    result_dir_test = os.path.join(result_dir, 'test')

    if not os.path.exists(result_dir_test):
        os.makedirs(os.path.join(result_dir_test, 'png', 'a2b'))
        os.makedirs(os.path.join(result_dir_test, 'png', 'b2a'))
        # os.makedirs(os.path.join(result_dir_test, 'numpy'))

    ## 네트워크 학습하기
    if mode == 'test':
        transform_test = transforms.Compose([Resize(shape=(ny, nx, nch)), Normalization(mean=MEAN, std=STD)])

        dataset_test_a = Dataset(data_dir=os.path.join(data_dir, 'test'), transform=transform_test, task=task,
                                 data_type='a')
        loader_test_a = DataLoader(dataset_test_a, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKER)

        # 그밖에 부수적인 variables 설정하기
        num_data_test_a = len(dataset_test_a)
        num_batch_test_a = np.ceil(num_data_test_a / batch_size)

        dataset_test_b = Dataset(data_dir=os.path.join(data_dir, 'test'), transform=transform_test, task=task,
                                 data_type='b')
        loader_test_b = DataLoader(dataset_test_b, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKER)

        # 그밖에 부수적인 variables 설정하기
        num_data_test_b = len(dataset_test_b)
        num_batch_test_b = np.ceil(num_data_test_b / batch_size)

    ## 네트워크 생성하기
    if network == "DeepEDN":
        netG_a2b = DeepEDN(in_channels=nch, out_channels=nch, nker=nker, norm=norm, nblk=18).to(device) #encription
        netG_b2a = DeepEDN(in_channels=nch, out_channels=nch, nker=nker, norm=norm, nblk=18).to(device) #decription

        netD_a = Discriminator(in_channels=nch, out_channels=1, nker=nker, norm=norm).to(device) #desciminator
        netROI = SimpleROI(in_channels=nch, out_channels=2048, nker=2*nker, norm=norm, nblk=4).to(device)

        init_weights(netG_a2b, init_type='normal', init_gain=0.02)
        init_weights(netG_b2a, init_type='normal', init_gain=0.02)
        init_weights(netD_a, init_type='normal', init_gain=0.02)
        init_weights(netROI, init_type='normal', init_gain=0.02) #no mention on the paper for ROI -did the same as cyclegan paper

    ## loss function definition
    fn_rec = nn.L1Loss().to(device) # reconstrucion loss : dislike cycle loss of cyclegan, we only need L1(F(G(x))-X)
    fn_dis = nn.BCELoss().to(device) # discriminator loss: same as gan loss of cyclegan -binary cross
    fn_enc = nn.BCELoss().to(device) # encription loss: log(1-D(G(x)))
    fn_roi = nn.MSELoss(reduction='mean').to(device) #ROI loss: Mean Square Error(generated cipher image - corresponding medical image)

    ## Optimizer
    optimG = torch.optim.Adam(itertools.chain(netG_a2b.parameters(), netG_b2a.parameters()), lr=lr, betas=(0.5, 0.999)) #encription & decription
    optimD = torch.optim.Adam(netD_a.parameters(), lr=lr, betas=(0.5, 0.999)) #Discriminator for A
    optimR = torch.optim.Adam(netROI.parameters(), lr=learning_rate) #ROI


    ## Additional functions
    fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
    fn_denorm = lambda x: (x * STD) + MEAN

    ## learing network
    st_epoch = 0

    # TRAIN MODE
    if mode == "test":
        netG_a2b, netG_b2a, \
        netD_a, netROI,\
        optimG, optimD, optimR, st_epoch = load(ckpt_dir=ckpt_dir,
                                        netG_a2b=netG_a2b, netG_b2a=netG_b2a, netROI = netROI,
                                        netD_a=netD_a, optimG=optimG, optimD=optimD, optimR=optimR)

        with torch.no_grad():
            netG_a2b.eval()
            netG_b2a.eval()

            #loss_ROI_test = []

            for batch, data in enumerate(loader_test_a, 1):
                # forward pass
                input_a = data['data_a'].to(device)

                output_b = netG_a2b(input_a)

                # Tensorboard 저장하기
                input_a = fn_tonumpy(fn_denorm(input_a))
                output_b = fn_tonumpy(fn_denorm(output_b))

                for j in range(input_a.shape[0]):
                    id = batch_size * (batch - 1) + j

                    input_a_ = input_a[j]
                    output_b_ = output_b[j]

                    input_a_ = np.clip(input_a_, a_min=0, a_max=1)
                    output_b_ = np.clip(output_b_, a_min=0, a_max=1)

                    plt.imsave(os.path.join(result_dir_test, 'png', 'a2b', '%04d_input_a.png' % id), input_a_)
                    plt.imsave(os.path.join(result_dir_test, 'png', 'a2b', '%04d_output_b.png' % id), output_b_)

                    print("TEST A: BATCH %04d / %04d | " % (id + 1, num_data_test_a))

'''

            for batch, data in enumerate(loader_test_b, 1):
                # forward pass
                input_b = data['data_b'].to(device)

                output_a = netG_b2a(input_b)

                # Tensorboard 저장하기
                input_b = fn_tonumpy(fn_denorm(input_b))
                output_a = fn_tonumpy(fn_denorm(output_a))

                for j in range(input_b.shape[0]):
                    id = batch_size * (batch - 1) + j

                    input_b_ = input_b[j]
                    output_a_ = output_a[j]

                    input_b_ = np.clip(input_b_, a_min=0, a_max=1)
                    output_a_ = np.clip(output_a_, a_min=0, a_max=1)

                    plt.imsave(os.path.join(result_dir_test, 'png', 'b2a', '%04d_input_b.png' % id), input_b_)
                    plt.imsave(os.path.join(result_dir_test, 'png', 'b2a', '%04d_output_a.png' % id), output_a_)

                    print("TEST B: BATCH %04d / %04d | " % (id + 1, num_data_test_b))

'''