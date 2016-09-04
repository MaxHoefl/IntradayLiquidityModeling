import pickle
import os
import pandas as pd
import numpy as np
import collections
import datetime
from matplotlib import pyplot as plt

from LobsterData import LOBSTER
from OTE import SARSA_LINAPPROX
from OTE_ac import SARSA_AC
from OTE_deepnet import SARSA_DEEP

ticker = 'AAPL'
starttime = '34200000'
endtime = '57600000'
nlevels = '100'
lobster_location = '/Volumes/INTENSO/LOBSTER/pickledLobster/v02/'


class OTE_ANALYZER():
    def __init__(self, TEST_TIMES_PATH):

        start_inventory = 1000
        ep_length = '3m'
        checkpoint_path = ""

        #ac_test_path = "/Users/mh/Documents/CSML/Masterarbeit/Python/RL_haven/AC/remote/July/test/"
        #deep_test_path = "/Users/mh/Documents/CSML/Masterarbeit/Python/RL_haven/DEEP/remote/test/"
        self.sarsa_AC = None
        self.sarsa_DEEP = None
        self.start_inventory = start_inventory
        self.checkpoint_path = "Null"
        self.testing_times = pickle.load(open(TEST_TIMES_PATH + "OTE_AC_testing_times", "rb"))
        self.lobster_location = '/Volumes/INTENSO/LOBSTER/pickledLobster/v02/'
        self.ep_length = ep_length
        self.means_ = None

    def _load_sarsa(self, mode, start_inventory, ep_length, CHECKPOINT_PATH):

        assert mode in ['ac', 'deep']

        if mode == 'ac':
            sarsa = SARSA_AC(start_inventory=start_inventory, ep_length=ep_length, checkpoint_path=CHECKPOINT_PATH)
        else:
            sarsa = SARSA_DEEP(start_intory=start_inventory, ep_length=ep_length, checkpoint_path=CHECKPOINT_PATH)

        #self.feature_dict = sarsa.feature_dict
        #self.action_dict = sarsa.action_dict
        self.sarsa = sarsa
        self.mode = mode

        # load weights
        if mode == 'ac':
            sarsa.pactor = pickle.load(open(CHECKPOINT_PATH + "OTE_AC_training_actor_params.p", "rb"))
            sarsa.pcritic = pickle.load(open(CHECKPOINT_PATH + "OTE_AC_training_critic_params.p", "rb"))
            #self.actor_params = pickle.load(open(CHECKPOINT_PATH + "OTE_AC_training_actor_params.p", "rb"))
            #self.critic_params = pickle.load(open(CHECKPOINT_PATH + "OTE_AC_training_critic_params.p", "rb"))
        else:
            sarsa.pactor = pickle.load(open(CHECKPOINT_PATH + "OTE_DEEP_training_actor_params.p", "rb"))
            sarsa.pcritic = pickle.load(open(CHECKPOINT_PATH + "OTE_DEEP_training_critic_params.p", "rb"))
            #self.actor_params = pickle.load(open(CHECKPOINT_PATH + "OTE_DEEP_training_actor_params.p", "rb"))
            #self.critic_params = pickle.load(open(CHECKPOINT_PATH + "OTE_DEEP_training_critic_params.p", "rb"))


    def _sample_obs(self, means_, remTime, remIntory, ASKq, bidask_state=None):

        assert isinstance(means_, dict)
        obs = pd.DataFrame(means_)
        obs["remTime"] = remTime
        obs["remIntory"] = remIntory
        obs["ASKq"] = ASKq
        if self.mode == 'deep':
            assert bidask_state is not None
            obs["bidask_state"] = bidask_state
        else:
            obs["bidask_state"] = 0
        col_locator = dict([(col_name, obs.columns.get_loc(col_name)) for col_name in obs.columns])
        obs = obs.values
        return obs.flatten(), col_locator


    def _feat2tile(self, obs, col_locator):
        state_vec = []
        sarsa = self.sarsa
        features = list(sarsa.feature_dict.keys())

        for feature in features:
            tiles = sarsa.feature_dict[feature] # -> get tiles for the current feature

            # Construct tile encoding
            if feature != "remTime":
                tile_vec = np.zeros((len(tiles)+1,)) # **
                feat_val = obs[col_locator[feature]]
                for ix, (fl, fu) in enumerate(tiles, start=0):
                    if (fl <= feat_val) & (feat_val < fu):
                        tile_vec[ix] = 1
                        break
                    elif tiles[-1][1] < feat_val:
                        tile_vec[len(tiles)] = 1
                        break
            else:
                tile_vec = np.zeros((len(tiles),)) # compare this to **, we dont have the len(tiles) plus 1 as the additional class is not there
                feat_val = obs[col_locator[feature]]
                #assert hasattr(feat_val, "seconds"), "feat_val must have attribute .seconds."
                #feat_val = int(feat_val.seconds)
                for ix, (fl, fu) in enumerate(tiles, start=0):
                    if feat_val == 0:
                        tile_vec[0] = 1
                    if (fl <= feat_val) & (feat_val < fu):
                        tile_vec[ix] = 1
                        break
                    elif tiles[-1][1] <= feat_val:
                        tile_vec[len(tiles)-1] = 1
                        break

            state_vec.extend(tile_vec)
        return np.array(state_vec)


    def _get_stateaction_vec(self, action_dict, state, action):
        # get one hot representation of action
        action_vec = self.sarsa.action_dict[action]
        # combine binary state vector and one hot action vector
        full_vec = np.concatenate((state,action_vec))
        return full_vec


    def _get_pi(self, state_vec):

        pi = []
        sarsa = self.sarsa

        for action in sarsa.ord_actiondict.keys():
            pi.append(np.dot(sarsa.pactor, sarsa._stateaction_vec(state=state_vec, action=action)))
            #state_action_vec = self._get_stateaction_vec(action_dict=action_dict, state=state_vec, action=action)
            #pi[action] = np.dot(actor_params, state_action_vec)

        return np.array(pi)

    def _get_feat_means(self):

        days = [day_ix[0] for day_ix in self.testing_times]
        stix = [day_ix[1] for day_ix in self.testing_times]

        df_times = pd.DataFrame({"Day": days, "Stix": stix})
        df_times.loc[:,"Day"] = pd.to_datetime(df_times.Day, format="%Y-%m-%d")
        df_times = df_times.sort_values(by="Day")
        df_times.loc[:,"Day"] = df_times.Day.astype(str)


        df_episodes = None
        day = "Null"

        for ix in df_times.index:
            if df_times.ix[ix, "Day"] != day:
                try:
                    day = df_times.ix[ix, "Day"]
                    fname = ticker + "_" + day + "_" + starttime + "_" + endtime + "_" + "lobsterclass_" + nlevels + ".pickle"
                    Lobster = pickle.load(open(lobster_location + fname, "rb"))
                    orderbook = Lobster.orderbook
                    orderbook["Spread"] = orderbook.ASKp1 - orderbook.BIDp1
                except (FileNotFoundError, IOError) as err:
                    print("File for day ", day, " not found.")
                    raise err

            start_ix = df_times.ix[ix, "Stix"]
            starting_time = orderbook["Timestamp"].iloc[start_ix]
            stopping_time = starting_time + pd.to_timedelta(self.ep_length)
            print("     ", starting_time)

            episode = orderbook.loc[(orderbook.Timestamp >= starting_time) &
                                    (orderbook.Timestamp <= stopping_time)]

            if df_episodes is not None:
                df_episodes = pd.concat([df_episodes, episode])
            else:
                df_episodes = episode


        AC_mean_features = ["Spread", "LastTradeDuration"]
        AC_mean_features.extend(["ASKp" + cl.__str__() for cl in range(1,11)])
        AC_mean_features.extend(["ASKs" + cl.__str__() for cl in range(1,11)])
        AC_mean_features.extend(["BIDp" + cl.__str__() for cl in range(1,11)])
        AC_mean_features.extend(["BIDs" + cl.__str__() for cl in range(1,11)])

        means_ = [(col, [df_episodes.loc[:, col].mean()]) for col in AC_mean_features]
        means_ = dict(means_)
        return means_


    def pipeline(self, remTime, remIntory, ASKq, bidask_state=None):

        # Stage 1: Get the means of the variables we need to stay constant
        self.means_ = pickle.load(open("means_.p", "rb"))
        """
        if self.means_ is None:
            self.means_ = self._get_feat_means()
            pickle.dump(self.means_, open("means_.p", "wb"))
        """
        # Stage 2: Get an observation and the column locator
        if self.mode == 'deep':
            assert bidask_state is not None
            obs, col_locator = self._sample_obs(means_=self.means_, remTime=remTime, remIntory=remIntory, ASKq=ASKq, bidask_state=bidask_state)

        else:
            obs, col_locator = self._sample_obs(means_=self.means_, remTime=remTime, remIntory=remIntory, ASKq=ASKq)

        # Stage 3: Transform the observation into a proper tile vector
        state_vec = self._feat2tile(obs=obs, col_locator=col_locator)

        # Stage 4: Get the probability distribution over the actions given the tile vector
        pi = self._get_pi(state_vec=state_vec)
        return pi


    def get_pi_dynamics(self, remTime_list, remIntory_list, ASKq_list):

        if self.mode == "deep":
            pi_dynamics = np.zeros((21, len(remTime_list), len(remIntory_list), len(ASKq_list), 2))
        else:
            pi_dynamics = np.zeros((21, len(remTime_list), len(remIntory_list), len(ASKq_list)))

        for time_ix, remTime in enumerate(remTime_list):
            for intory_ix, remIntory in enumerate(remIntory_list):
                for q_ix, ASKq in enumerate(ASKq_list):
                    print("remTime: ", remTime, " remIntory: ", remIntory, " ASKq:", ASKq)
                    if self.mode == "deep":
                        pi_dynamics[:,time_ix, intory_ix, q_ix, 0] = self.pipeline(remTime=remTime, remIntory=remIntory, ASKq=ASKq, bidask_state=0)
                        pi_dynamics[:,time_ix, intory_ix, q_ix, 1] = self.pipeline(remTime=remTime, remIntory=remIntory, ASKq=ASKq, bidask_state=1)
                    else:
                        tmp = self.pipeline(remTime=remTime, remIntory=remIntory, ASKq=ASKq)
                        pi_dynamics[:,time_ix, intory_ix, q_ix] = tmp
                        #pi_dynamics[:,time_ix, intory_ix, q_ix] = self.pipeline(remTime=remTime, remIntory=remIntory, ASKq=ASKq)


        pickle.dump(pi_dynamics, open("DEEP_pi_dynamics.p", "wb"))



if __name__ == "__main__":

    TEST_TIMES_PATH = "/Users/mh/Documents/CSML/Masterarbeit/Python/RL_haven/AC/test/"
    CHECKPOINT_PATH = "/Users/mh/Documents/CSML/Masterarbeit/Python/RL_haven/DEEP/"

    ep_length = pd.to_timedelta('3m')
    start_intory = 1000
    #remtime_tiles = [((k-1)/30 * ep_length.seconds, k/30 * ep_length.seconds) for k in range(1,31)]
    #remintory_tiles = [(((k-1)/10.0) * start_intory, (k/10.0) * start_intory) for k in range(1,10)]
    #askqueue_tiles = [(k, k+1) for k in range(1,10)] # Dont chagne this


    remTime_list = [10,60,120,170]
    remIntory_list = [10,200,500,700]
    ASKq_list = [1,2,4,8]

    Analyzer = OTE_ANALYZER(TEST_TIMES_PATH=TEST_TIMES_PATH)

    Analyzer._load_sarsa(mode='deep', start_inventory=Analyzer.start_inventory, ep_length=Analyzer.ep_length,
                         CHECKPOINT_PATH=CHECKPOINT_PATH)

    Analyzer.get_pi_dynamics(remTime_list=remTime_list, remIntory_list=remIntory_list, ASKq_list=ASKq_list)

