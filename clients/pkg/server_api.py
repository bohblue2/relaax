from __future__ import print_function

import base64
import io
import json
import numpy
import sys
import socketIO_client


class _NDArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            output = io.BytesIO()
            numpy.savez_compressed(output, obj=obj)
            return {'b64npz': base64.b64encode(output.getvalue())}
        return json.JSONEncoder.default(self, obj)


class ServerAPI(socketIO_client.LoggingNamespace):
    def __init__(self, game, *args, **kwargs):
        socketIO_client.LoggingNamespace.__init__(self, *args, **kwargs)
        self._game = game
        self._game_played = 0

    def on_connected(self, *args):
        print('on_connected')
        self.emit('act', self._dump_state())

    def on_act(self, action):
        reward, reset = self._game.act(action)
        if reset:
            self.emit('reward_and_reset', reward)
        else:
            self.emit('reward_and_act', reward, self._dump_state())

    def on_reset(self, score):
        self._game_played += 1
        print("Score at game", self._game_played, "=", score)
        self._game.reset()
        self.emit('act', self._dump_state())

    def _dump_state(self):
        return json.dumps(self._game.state(), cls=_NDArrayEncoder)
