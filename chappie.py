from __future__ import division
from __future__ import print_function
from vizdoom import *
from random import sample, randint, random
import numpy as np
import skimage.color, skimage.transform
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import math
from PIL import Image
import matplotlib.pyplot as plt


# Some basic options - tickrate is how many frames to process, higher values are faster at the
# cost of accuracy. Resolution should be left alone for now, and config file path is used to
# choose which scenario we use. Only proven to converge on simpler_basic for now.

tickrate = 5
resolution = (64,64)
config_file_path = "simpler_basic.cfg"


def preprocess(img):
    img = skimage.transform.resize(img, resolution)
    img = img.astype(np.float32)
    return img

unloader = transforms.ToPILImage()  # reconvert into PIL image

plt.ion()

def imshow(tensor, title=None, figure=0):
    plt.figure(figure)
    image = tensor.clone().cpu()  # we clone the tensor to not do changes on it
    image = image.view(3, 64, 64)  # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001) # pause a bit so that plots are updated

# despite the name, this is a simple feedforward network not a VAE.
# name is a left-over from previous ideas being tested.

class ConvVAE(nn.Module):
    def __init__(self, n_actions):
        super(ConvVAE, self).__init__()
        self.conv1 = nn.Conv2d(3, 24, 4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(24, 48, 4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(48, 96, 4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(96, 192, 4, stride=2, padding=1)
        self.fc1 = nn.Linear(192*4*4, n_actions)

    def encode(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = h.view(-1,192*4*4)
        h = F.sigmoid(self.fc1(h))
        return h

    def forward(self, x):
        return self.encode(x)

# set up our doom environment. 

def initialize_vizdoom(config_file_path):
    print("Initializing doom...")
    game = DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(True)
    game.set_mode(Mode.PLAYER)
    game.set_screen_format(ScreenFormat.RGB24)
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    game.init()
    print("Doom initialized.")
    return game

loss_act = nn.BCELoss()

# time to run this fucker!

if __name__ == '__main__':
    game = initialize_vizdoom(config_file_path)
    n = game.get_available_buttons_size()
    vae = ConvVAE(n)
    optimizer = torch.optim.Adam(vae.parameters(), 1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience = 1000, verbose=True)
    step = 0
    score = 0
    avg_score = None
    avg_score2 = None
    avg_act = torch.ones(1,n).mul(0.5)
    act = torch.rand(1,n).mul(0).add(0.5)
    poop = act.detach().round()
    grad_buff = [None,None,None,None,None,None,None,None,None,None]
    game.new_episode()
    while True:
        if game.is_episode_finished():
            if avg_score is None:
                avg_score = -250
                avg_score2 = avg_score
            avg_score = avg_score * 0.99 + game.get_total_reward() * 0.01
            avg_score2 = avg_score2 * 0.9 + game.get_total_reward() * 0.1
            print(avg_score, avg_score2, avg_score2-avg_score)
            grad_buff = [None,None,None,None,None,None,None,None,None,None]
            scheduler.step(avg_score)
            game.new_episode()
            
        s1 = preprocess(game.get_state().screen_buffer)
        s1 = s1.reshape([1,resolution[0],resolution[1],3])
        s1 = torch.from_numpy(s1)
        s1 = s1.transpose(1,2).transpose(1,3)

        act = vae.forward(s1)
        
        # throwing in some noise makes the policy network stochastic, which helps with
        # exploration - the net works off probabilities, not certainties.
        
        actions = act.add(torch.rand_like(act).add(-0.5).mul(0.5)).clamp(0,1)
        poop = actions.round().detach()
        poopact = poop.numpy().tolist()[0]
        game.make_action(poopact,tickrate)
        score = game.get_last_reward()
        loss_enc = loss_act(actions, poop)
        loss_enc.backward()
        iterator = 0 

        # the magic of chappie is here. Instead of having a replay memory buffer or a
        # critic network, we store the action history in the gradients themselves. With
        # this trick we can update the parameters every step, rather than every episode,
        # and it uses far less resources than PPO style optimization.

        for param in vae.parameters():
            if grad_buff[iterator] is None:
                grad_buff[iterator] = param.grad.clone().detach()
            else:
                grad_buff[iterator] = grad_buff[iterator].mul(0.95).add(param.grad).detach()
            param.grad = grad_buff[iterator].mul((score)*0.0003).detach()
            iterator = iterator+1

        avg_act.mul_(0.5).add_(act.mul(0.5))
        optimizer.step()
        optimizer.zero_grad()

        step = step+1
