################################################################################################################
# This class will implement a Sarsa(lambda) algorithm based on state-representations by a trained LSTM
# The LSTM was trained on mini-episodes of length 10 from the orderbook (40 features included) and was trained
# to predict if the next event is a marketorder.
# For an incoming observation, we take this observations with its preceding 9 observations, form a (1,10,40) mini-episode,
# pass it through the LSTM and choose the hidden layer before the last sigmoid layer as representation for the current
# state.
# We then search for the nearest neighbor of those clusters we have already found that way.
# The state will be whatever state the embedded observation is closest to.
#################################################################################################################


import numpy as np
import sys
import time
import collections

from OTE import SARSA_LINAPPROX


class SARSA_DEEP(SARSA_LINAPPROX):

    def __init__(self, ep_length, start_intory, checkpoint_path):

        super(SARSA_DEEP, self).__init__(start_intory=start_intory, ep_length=ep_length, checkpoint_path=checkpoint_path)


        # onehot encoding for cluster state
        bidask_tiles = [(0,1),(1,2)]

        tiles = (("remTime",          self.remtime_tiles),
                ("remIntory",         self.remintory_tiles),
                ("ASKq",              self.askqueue_tiles),
                ("Spread",            self.spread_tiles),
                ("bidask_state",      bidask_tiles),
                ("LastTradeDuration", self.tiles_LTD))

        self.feature_dict = collections.OrderedDict(tiles)
        self.nFeatures = np.sum(np.asarray([len(self.feature_dict[key])+1 if (key!="remTime")&(key!="remIntory")
                                                                          else len(self.feature_dict[key])
                                                                          for key in self.feature_dict.keys()]))




if __name__ == "__main__":

    if len(sys.argv) == 1:
        ep_length = '3m'
        n_episodes = '50'
        start_inventory=1000
        lobster_location = "/Volumes/INTENSO/LOBSTER/pickledLobster/v02/"
        lstm_location = "/Users/mh/Documents/CSML/Masterarbeit/Python/LSTM_v2/m04/"
        checkpoint_path = "/Users/mh/Documents/CSML/Masterarbeit/Python/RL_haven/DEEP/remote/test/"
        replay_path = "/Users/mh/Documents/CSML/Masterarbeit/Python/RL_haven/DEEP/remote/test/"


    else:
        try:
            ep_length, start_inventory, n_episodes, lobster_location, checkpoint_path = sys.argv[1:]
        except IndexError:
            print("Usage: python3 OTE_deepnet.py\n<ep_length>\n<start_inventory>\n\n<n_episodes>\n<lobster_location>\n\n<checkpoint_path>\n<replay_path>")
            sys.exit(1)

    start_inventory = int(start_inventory)

    starttime = time.time()

    sarsa = SARSA_DEEP(ep_length=ep_length, start_intory=start_inventory, checkpoint_path=checkpoint_path)
    crewards, save_times = sarsa.learn(nEpisodes=int(n_episodes),
                                       lobster_location=lobster_location,
                                       replay_path=replay_path,
                                       freeze=False, sit_leave=False)


    sarsa.save_model(cum_rewards=crewards,
                     save_times=save_times,
                     actor_stepsize=sarsa.pactor_stepsize,
                     critic_stepsize=sarsa.pcritic_stepsize,
                     replay_path=replay_path)

    endtime = time.time()
    print("time elapsed: ", endtime - starttime)
















