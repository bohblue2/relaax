import sys
import numpy as np
from scipy.misc import imresize
from ale_python_interface import ALEInterface

from params import ROM


class GameState(object):
    def __init__(self, rand_seed, display=False, no_op_max=7):
        self.ale = ALEInterface()
        self.ale.setInt('random_seed', rand_seed)
        self._no_op_max = no_op_max

        if display:
            self._setup_display()

        self.ale.loadROM(ROM)

        # collect minimal action set
        self.real_actions = self.ale.getMinimalActionSet()

        # height=210, width=160
        self._screen = np.empty((210, 160, 1), dtype=np.uint8)

        self.reset()

    def _process_frame(self, action, reshape):
        reward = self.ale.act(action)
        terminal = self.ale.game_over()

        # screen shape is (210, 160, 1)
        self.ale.getScreenGrayscale(self._screen)

        # reshape it into (210, 160)
        reshaped_screen = np.reshape(self._screen, (210, 160))

        # resize to height=110, width=84
        resized_screen = imresize(reshaped_screen, (110, 84))

        x_t = resized_screen[18:102, :]
        if reshape:
            x_t = np.reshape(x_t, (84, 84, 1))
        x_t = x_t.astype(np.float32)
        x_t *= (1.0 / 255.0)
        return reward, terminal, x_t

    def _setup_display(self):
        if sys.platform == 'darwin':
            import pygame
            pygame.init()
            self.ale.setBool('sound', False)
        elif sys.platform.startswith('linux'):
            self.ale.setBool('sound', True)
        self.ale.setBool('display_screen', True)

    def reset(self):
        self.ale.reset_game()

        # randomize initial state
        if self._no_op_max > 0:
            no_op = np.random.randint(0, self._no_op_max + 1)
            for _ in range(no_op):
                self.ale.act(0)

        _, _, x_t = self._process_frame(0, False)

        self.reward = 0
        self.terminal = False
        self.s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

    def process(self, action):
        # convert original 18 action index to minimal action set index
        real_action = self.real_actions[action]

        r, t, x_t1 = self._process_frame(real_action, True)

        self.reward = r
        self.terminal = t
        self.s_t1 = np.append(self.s_t[:, :, 1:], x_t1, axis=2)

    def update(self):
        self.s_t = self.s_t1
