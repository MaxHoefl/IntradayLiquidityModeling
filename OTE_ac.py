import numpy as np
import collections
import pickle
import copy as cp
import pandas as pd
import datetime
import re
import time
import sys

from OTE import SARSA_LINAPPROX

class SARSA_AC(SARSA_LINAPPROX):

    def __init__(self, ep_length, start_inventory, checkpoint_path):

        ### Initialize all the stuff from SARSA_LINAPPROX
        super(SARSA_AC, self).__init__(start_intory=start_inventory,
                                       ep_length=ep_length,
                                       checkpoint_path=checkpoint_path)


        # Tiles for ask sizes
        asks_tiles = [(0,25),(17,34),(25,83),(34,105),(105,250),(250,625)] # Dont change this
        # Tiles for ask prices: We create a dictionary with keys: Levels, values: Quintiles of ask prices on that level
        try:
            df_perc_ASKp = pickle.load(open("df_perc_ASKp.p", "rb"))
            tiles_ASKp = {}
            for col in df_perc_ASKp.columns:
                tmp = df_perc_ASKp.loc[:,[col]]
                tiles_ASKp[col] = [(tmp.iloc[k].values.tolist()[0], tmp.iloc[k+1].values.tolist()[0]) for k in range(len(tmp)-1)]
        except IOError as io:
            print("df_perc.p not found.")
            raise io


        tiles = ( ("remTime",self.remtime_tiles),
                ("remIntory",self.remintory_tiles),
                ("ASKq",self.askqueue_tiles),
                ("ASKs1", asks_tiles),
                ("ASKs2", asks_tiles),
                ("ASKs3", asks_tiles),
                ("ASKs4", asks_tiles),
                ("ASKs5", asks_tiles),
                ("ASKs6", asks_tiles),
                ("ASKs7", asks_tiles),
                ("ASKs8", asks_tiles),
                ("ASKs9", asks_tiles),
                ("ASKs10", asks_tiles),
                ("ASKp1", tiles_ASKp["Lev1"]),
                ("ASKp2", tiles_ASKp["Lev2"]),
                ("ASKp3", tiles_ASKp["Lev3"]),
                ("ASKp4", tiles_ASKp["Lev4"]),
                ("ASKp5", tiles_ASKp["Lev5"]),
                ("ASKp6", tiles_ASKp["Lev6"]),
                ("ASKp7", tiles_ASKp["Lev7"]),
                ("ASKp8", tiles_ASKp["Lev8"]),
                ("ASKp9", tiles_ASKp["Lev9"]),
                ("ASKp10", tiles_ASKp["Lev10"]),
                ("Spread", self.spread_tiles),
                ("LastTradeDuration", self.tiles_LTD))


        self.feature_dict = collections.OrderedDict(tiles)
        self.nFeatures = np.sum(np.asarray([len(self.feature_dict[key])+1 if (key!="remTime")&(key!="remIntory")
                                                                          else len(self.feature_dict[key])
                                                                          for key in self.feature_dict.keys()]))


if __name__ == '__main__':

    if len(sys.argv) == 1:
        start_inventory = 1000
        ep_length = '3m'
        n_episodes = '25'
        lobster_location = "/Volumes/INTENSO/LOBSTER/pickledLobster/v02/"
        checkpoint_path = "/Users/mh/Documents/CSML/Masterarbeit/Python/RL_haven/AC/"
        replay_path = "/Users/mh/Documents/CSML/Masterarbeit/Python/RL_haven/AC/"
        #replay_path = None
    else:
        try:
            ep_length, start_inventory, n_episodes, lobster_location, checkpoint_path = sys.argv[1:]
        except IndexError:
            print("Usage: python3 OTE_ac.py\n<ep_length>\n<start_inventory>\n<day>\n<n_episodes>\n<lobster_location>\n<checkpoint_path>\n<replay_path>")
            sys.exit(1)

    starttime = time.time()

    start_inventory = int(start_inventory)

    sarsa = SARSA_AC(start_inventory=start_inventory, ep_length=ep_length, checkpoint_path=checkpoint_path)

    crewards, save_times = sarsa.learn(nEpisodes=int(n_episodes),
                                       lobster_location=lobster_location,
                                       replay_path=replay_path,
                                       freeze=False, sit_leave=False)

    sarsa.save_model(cum_rewards=crewards, save_times=save_times,
                     actor_stepsize=sarsa.pactor_stepsize,
                     critic_stepsize=sarsa.pcritic_stepsize,
                     replay_path=replay_path)

    endtime = time.time()
    print("time elapsed: ", endtime - starttime)


