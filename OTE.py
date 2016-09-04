########################################################################
# This is a base class for an actor critic implementation to deal with
# optimal order execution.
# The features are held simple and are supposed to be augmented by
# child classes.
########################################################################


import numpy as np
import collections
import pickle
import copy as cp
import pandas as pd
import re
import time
import sys
import os
import datetime


class SARSA_LINAPPROX():

    def __init__(self, checkpoint_path, start_intory=3000, ep_length='5m'):
        """
        This class is for performing SARSA(lambda) actor-critic model
        Description of features used:
        - remtime: Remaining time for submitting limit orders
        - remintory: Remaining inventory of stocks
        - askqueue: Number of levels in front of agent
        - asks: Size at each ask level
        - askp: Price at each ask level
        - duration: Time passed since last market order
        :param start_intory: the inventory that needs to be executed
        :param ep_length: length of an episode as string (e.g. 1m -> 1 minute, 1s -> 1 second etc...)
        :return:
        """

        #self.dates = ["2014-05-01", "2014-05-02", "2014-05-05", "2014-05-06", "2014-05-07",
        #              "2014-05-08", "2014-05-09", "2014-05-12", "2014-05-13", "2014-05-14",
        #              "2014-05-15", "2014-05-16", "2014-05-19", "2014-05-20", "2014-05-21",
        #              "2014-05-22", "2014-05-23", "2014-05-27", "2014-05-28", "2014-05-29",
        #              "2014-05-30"]
        #self.dates = ["2014-07-01","2014-07-02", "2014-07-03", "2014-07-07", "2014-07-08",
        #              "2014-07-09", "2014-07-10", "2014-07-11", "2014-07-14", "2014-07-15",
        #              "2014-07-16", "2014-07-17", "2014-07-18", "2014-07-21", "2014-07-22",
        #              "2014-07-23"]
        self.dates = ["2014-07-24", "2014-07-25", "2014-07-28", "2014-07-29", "2014-07-30", "2014-07-31"]

        # Model weights
        self.pcritic = None
        self.pactor = None
        self.pactor_stepsize = 1 # Step size for critic update.
        self.pcritic_stepsize = 1 # Step size for actor update.
        self.checkpoint_path = None

        if checkpoint_path is not None:
            self.checkpoint_path = checkpoint_path
            try:
                self.pcritic = pickle.load(open(checkpoint_path + "OTE_critic_params.p", "rb"))
                self.pactor = pickle.load(open(checkpoint_path + "OTE_actor_params.p", "rb"))
                self.pactor_stepsize = pickle.load(open(checkpoint_path + "OTE_actor_stepsize.p", "rb"))
                self.pcritic_stepsize = pickle.load(open(checkpoint_path + "OTE_critic_stepsize.p", "rb"))
                print("Checkpoint found at ", checkpoint_path)
                print("Continuing training...")
            except (FileNotFoundError, IOError, TypeError) as err:
                print("No checkpoint found at ", checkpoint_path)
                print("Starting training...")

        self.ep_length = pd.to_timedelta(ep_length) # length of episode
        ep_length = self.ep_length

        # Tile construction
        assert hasattr(ep_length, "seconds"), "ep_length must be timedelta."
        self.remtime_tiles = [((k-1)/30 * ep_length.seconds, k/30 * ep_length.seconds) for k in range(1,31)]
        self.remintory_tiles = [(((k-1)/10.0) * start_intory, (k/10.0) * start_intory) for k in range(1,10)]
        self.askqueue_tiles = [(k, k+1) for k in range(1,10)] # Dont chagne this
        self.spread_tiles = [(0,0.01),(0.01,0.02),(0.02,0.03)] # Dont change this

        # Tiles for the ask prices: We create a dictionary with keys: Levels, values: Quintiles of ask prices on that level
        try:
            df_perc_LTD = pickle.load(open("df_perc_LastTradeDuration.p","rb"))
            self.tiles_LTD = [(df_perc_LTD.iloc[k].values.tolist()[0], df_perc_LTD.iloc[k+1].values.tolist()[0]) for k in range(len(df_perc_LTD)-1)]
        except IOError as io:
            print("df_perc_LastTradeDuration.p not found")
            raise io

        tiles = (("remTime",          self.remtime_tiles),
                ("remIntory",         self.remintory_tiles),
                ("ASKq",              self.askqueue_tiles),
                ("Spread",            self.spread_tiles),
                ("LastTradeDuration", self.tiles_LTD))

        self.feature_dict = collections.OrderedDict(tiles)
        self.nFeatures = np.sum(np.asarray([len(self.feature_dict[key])+1 if (key!="remTime")&(key!="remIntory")
                                                                          else len(self.feature_dict[key])
                                                                          for key in self.feature_dict.keys()]))

        self.start_intory = cp.deepcopy(start_intory)
        self.rem_intory = start_intory
        self.rem_time = None
        self.end_time = None
        self.curr_level = 0
        self.curr_price = 0
        self.curr_time = None
        self.askqueue_vol = 0 # records how much volume is in front of the agent

        self.marketorders = None
        self.limitorders = None

        self.gamma = 1 # Discount factor
        self.lam = 1 # Parameter for eligibility traces

        ### What is the length of the tile vector: Note that although we might have k intervals for feature j, we need
        # k+1 entries for feature j in the tile vector as the last case always is that the feature value is greater than the
        # largest upper bound among the intervals
        self.nLevels = 10
        self.terminal = False

        # one hot encoding for actions i1, ... , i10, q1, ... , q10, idle
        self.init_actionset = ["m"+l.__str__() for l in range(1,self.nLevels+1)] + ["q"+l.__str__() for l in range(1,self.nLevels+1)]
        self.actionset = self.init_actionset + ['idle']
        #self.actionset = ["m"+l.__str__() for l in range(1,self.nLevels+1)] + ["q"+l.__str__() for l in range(1,self.nLevels+1)] + ["idle"]

        self.nActions = len(self.actionset)
        self.action_dict = dict([(action_key, np.zeros((self.nActions,))) for action_key in self.actionset])
        counter = 0
        for key in self.action_dict.keys():
            self.action_dict[key][counter] = 1 # set unique 1 in one-hot action vector
            counter += 1

        self.ord_actiondict = collections.OrderedDict()
        for act in self.actionset:
            self.ord_actiondict[act] = self.action_dict[act]

    def _feat2tile(self, obs):
        """ WORKS
        - feat2tile - This function converts a feature vector into a binary tile vector
        :param obs: array of current row in orderbook
        :return:
        """

        col_locator = self.col_locator
        obs[col_locator["remTime"]] = self.end_time - self.curr_time
        obs[col_locator["remIntory"]] = self.rem_intory
        obs[col_locator["ASKq"]] = self.curr_level

        # Transform feature vector into binary tile vector
        features = [feat for feat in self.feature_dict.keys()]
        #obs = obs[[col_locator[feat] for feat in features]]
        state_vec = []

        for feature in features:

            tiles = self.feature_dict[feature] # -> get tiles for the current feature

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
                assert hasattr(feat_val, "seconds"), "feat_val must have attribute .seconds."
                feat_val = int(feat_val.seconds)
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

        return np.asarray(state_vec)

    def load_data(self, day, lobster_location):
        """
        Loads in the limit order book data and creates training episodes
        :return:
        """
        ticker = 'AAPL'
        starttime = '34200000'
        endtime = '57600000'
        nlevels = '100'
        fname = ticker + "_" + day + "_" + starttime + "_" + endtime + "_" + "lobsterclass_" + nlevels + ".pickle"

        Lobster = pickle.load(open(lobster_location + fname, "rb"))


        orderbook = Lobster.orderbook
        orderbook.loc[orderbook.lo_level == -1, "lo_level"] = self.nLevels + 1
        orderbook["Spread"] = orderbook.ASKp1 - orderbook.BIDp1
        orderbook["Midprice"] = orderbook.BIDp1 + (orderbook.ASKp1 - orderbook.BIDp1)/2
        orderbook["remTime"] = 0
        orderbook["remIntory"] = 0
        orderbook["ASKq"] = 0
        orderbook = orderbook.loc[~orderbook.index.duplicated()] # the duplicates are due to marketorders that pierce through several levels and hence generate
                                                                                # multiple market events
        orderbook = orderbook.reset_index(drop=True)

        self.orderbook = orderbook
        self.limitorders = Lobster.limitorders
        self.marketorders = Lobster.marketorders

        # The following dictionary is important: It allows us to work only with numpy arrays by indexing via self.col_locator
        self.col_locator = dict([(col_name, self.orderbook.columns.get_loc(col_name)) for col_name in self.orderbook.columns])

    def _step(self,obs):
        """
        - step - This function returns a new state and a reward
        :return:
        """
        # Get (new) feature vector
        state_vec = self._feat2tile(obs=obs)
        col_locator = self.col_locator
        # Process the information of the new market event
        reward = 0


        try:
            # Check if limit order arrived and update currLevel/askqueue_vol if necessary
            if (obs[col_locator["Type"]] == 1) & (obs[col_locator["lo_level"]] < self.curr_level):
                self.curr_level += 1
                self.askqueue_vol += obs[col_locator["Size"]]

            # Check if order was completely cancelled and removes volume in front of us (and updates our level)
            elif (obs[col_locator["Type"]] == 3) & (obs[col_locator["lo_level"]] <= self.curr_level) & (obs[col_locator["OrderID"]] <= self.curr_id):
                self.askqueue_vol -= obs[col_locator["Size"]]
                if self.curr_level != 1:
                    self.curr_level -= 1

            # Check if order was partially cancelled and removes volume in front of us
            elif (obs[col_locator["Type"]] == 2) & (obs[col_locator["lo_level"]] <= self.curr_level) & (obs[col_locator["OrderID"]] <= self.curr_id):

                self.askqueue_vol -= obs[col_locator["Size"]]

            # Check if (hidden) marketorder arrived on the ask side
            elif ((obs[col_locator["Type"]] == 4) | (obs[col_locator["Type"]] == 5)) & (obs[col_locator["TradeDirection"]] == 1):
                reward = self._get_reward(obs)

        except ValueError as ve:
            print(obs)
            raise ve

        return state_vec, reward

    def _get_reward(self, obs):
        """
        The reward is the proceeds of our limit order (possibly 0) versus the proceeds of the best placed order (highest possible proceeds).
        To compute the highest possible proceeds we compute the proceeds on every level and then take the max
        :param obs:
        :return:
        """
        col_locator = self.col_locator
        # Check if we have no time left, execute everything as market order. The "reward" is the implementation shortfall
        # i.e. the difference between the best bid time the executed volume minus the actual realized prices * volumes
        # at each bid layer.
        if self.terminal:
            """
            realized_proceeds = 0
            bid_level = 0
            # while there is inventory left, eat into the next bid layer
            while self.rem_intory > 0:
                bid_level += 1
                if self.rem_intory < feat_vec.iloc[0,feat_vec.columns.get_loc("BIDp"+ bid_level.__str__())]:
                    realized_proceeds += self.rem_intory * feat_vec.iloc[0,feat_vec.columns.get_loc("BIDp"+ bid_level.__str__())]
                    self.rem_intory -= feat_vec.iloc[0,feat_vec.columns.get_loc("BIDs"+ bid_level.__str__())]
                else:
                    realized_proceeds += feat_vec.iloc[0,feat_vec.columns.get_loc("BIDs"+ bid_level.__str__())] * feat_vec.iloc[0,feat_vec.columns.get_loc("BIDp"+ bid_level.__str__())]
                    self.rem_intory -= feat_vec.iloc[0,feat_vec.columns.get_loc("BIDs"+ bid_level.__str__())]

            return realized_proceeds - self.rem_intory * feat_vec.iloc[0,feat_vec.columns.get_loc("BIDp1")]
            """
            return -self.rem_intory * 0.0035 # trading agent has to pay broker commission of 0.0035 cents per share on NASDAQ for
                                             # liquidating the remaining stocks


        # market order size
        mo_size = self.marketorders.loc[self.marketorders.Timestamp == obs[col_locator["Timestamp"]], "Size"].iloc[0]

        # best_proceeds will be the highest possible payoff from market order execution if one could choose level of limitorder freely
        best_proceeds = 0

        # marketorder does not change the price
        if mo_size < obs[col_locator["ASKs1"]]:
            best_proceeds = obs[col_locator["ASKp1"]] * mo_size

        # marketorder changes the price
        else:
            ordersize = cp.deepcopy(mo_size)
            counter = 0
            while (ordersize > 0) & (counter <= self.nLevels - 1):
                counter += 1
                # find highest proceeds by going through the levels and compute the residue market order size times the price on that level
                if obs[col_locator["ASKp" + counter.__str__()]] * ordersize > best_proceeds:
                    best_proceeds = obs[col_locator["ASKp" + counter.__str__()]] * ordersize
                ordersize -= obs[col_locator["ASKs" + counter.__str__()]]

            # Update current level: Counter gives how many levels have been eaten up by the marketorder
            self.curr_level -= counter

        actual_proceeds = np.maximum((mo_size - self.askqueue_vol),0) * self.curr_price

        # Update self.curr_level, self.askqueue_vol after marketorder has been processed
        exec_vol = np.maximum((mo_size - self.askqueue_vol),0)
        self.rem_intory -= exec_vol

        if self.askqueue_vol > mo_size:
            self.askqueue_vol -= mo_size

        else:
            self.askqueue_vol = 0
            self.curr_level = 1

        return actual_proceeds - best_proceeds

    def _stateaction_vec(self, state, action):
        assert isinstance(state, np.ndarray), "state is not a numpy array"
        assert isinstance(action, str), "action is not a string"
        # get one hot representation of action
        action_vec = self.action_dict[action]
        # combine binary state vector and one hot action vector
        full_vec = np.concatenate((state,action_vec))

        return full_vec

    def _softmax(self,state_vec, action):
        # returns pi(s,a)
        full_vec = self._stateaction_vec(state_vec, action)
        denominator = 0
        # Compute denominator
        for act in self.actionset:
            vec = self._stateaction_vec(state_vec, act)
            #vec = np.concatenate((state_vec,self.action_dict[act]))
            #vec = state_vec.extend(self.action_dict[act])
            denominator += np.exp(np.dot(vec, self.pactor))

        return np.exp(np.dot(full_vec, self.pactor)) / denominator

    def _ObsGenerator(self, starting_time):

        orderbook = self.orderbook

        ep_length = self.ep_length
        ending_time = starting_time + ep_length

        ### Generator gives next line of orderbook until episode is over
        episode = orderbook.loc[(orderbook.Timestamp >= starting_time) &
                                (orderbook.Timestamp <= ending_time)]
        initMidprice = episode["Midprice"].iloc[0]
        pcols = [col for col in episode.columns if col.startswith("ASKp") | col.startswith("BIDp")]
        episode.loc[:,pcols] = episode.loc[:,pcols] / initMidprice
        episode = episode.reset_index(drop=True)
        print("episode length: ", episode.shape[0])

        currindex = -1
        while currindex < episode.shape[0] - 1:
            currindex += 1
            #self.currindex = currindex
            #print(currindex, episode["Type"].iloc[currindex], self.rem_intory, (self.end_time - self.curr_time).seconds)
            yield episode.iloc[currindex].values

    def _choose_action(self, obs, new_state=None, plist=None):
        """
        Chooses action either uniformly from the actionset or according to the agent (which is controlled by self.pcritic)
        :param obs:
        :param new_state:
        :param plist:
        :return:
        """

        nActions = self.nActions
        actionset = self.actionset
        init_actionset = self.init_actionset
        init_nActions = len(init_actionset)
        col_locator = self.col_locator
        midprice = obs[col_locator["Midprice"]]
        askp1 = obs[col_locator["ASKp1"]]
        bidp1 = obs[col_locator["BIDp1"]]

        while True:
            if plist is None:
                #np.random.seed(1337)
                act_ix = np.random.choice(np.arange(0, init_nActions))
                action = init_actionset[act_ix]
            else:
                assert new_state is not None, "_choose_action(): If you wanna take an action wrt pi, pass the state as well."
                # np.random.seed(1337)
                act_ix = np.random.choice(np.arange(0,self.nActions), p=plist)
                action = actionset[act_ix]

            if (action != "idle"):
                lev = int(re.findall(r'\d+', action)[0])
                # If act == "i1" check if bid and ask price are far enough apart
                if action == "m1":
                    if (askp1 * midprice - bidp1 * midprice > 0.01):
                        break
                    # It can happen that agent doesn't want to do anything but improve level 1. However if the spread is
                    # too narrow, we don't allow that. In that case he stays idle.
                    else:
                        return "q1"


                # If act == "ix" x > 1, check if ASKpx and ASKp(x-1) are far enough apart
                elif (action == "m") & (lev != 1):
                    # if action is improvement then check if the lower level price is at least 0.01 apart
                    if (obs[col_locator["ASKp" + lev.__str__()]] * midprice - obs[col_locator["ASKp" + (lev-1).__str__()]] * midprice > 0.01):
                        break

                # If act == "qx", action is valid (we assume first 10 levels are always populated)
                else:
                    break
            else:
                break

        return action

    def _get_new_obs(self, gen):
        """
        We just have this in a different function so we can override more easily in children's classes
        :param gen: Generator (from self._ObsGenerator(...))
        :return: next observation (numpy array of current row in orderbook)
        """
        return gen.__next__()

    def _initialize_weights(self, stateaction_vec, state_vec):
        ### Initialize model parameters if they dont exist yet
        if self.pactor is None:
            #np.random.seed(1337)
            self.pactor = np.random.rand(len(stateaction_vec),)
        if self.pcritic is None:
            #np.random.seed(1337)
            self.pcritic = np.random.rand(len(state_vec),)

    def _housekeeping(self, new_action, initMidprice, obs):
        # if new action is an improvement
        col_locator = self.col_locator

        if new_action.startswith("m"):
            lev = re.findall(r'\d+',new_action)[0]
            self.curr_price = obs[col_locator["ASKp" + lev]] - 0.01/initMidprice
            self.curr_level = int(new_action[-1])
            asks_cols = [col for col in self.orderbook.columns if col.startswith("ASKs") if (int(re.findall(r'\d+', col)[0]) < self.curr_level)]
            self.askqueue_vol = self.askqueue_vol = np.sum(obs[[col_locator[col] for col in asks_cols]])
            self.curr_id = obs[col_locator["OrderID"]]

        # if new action is sitting on top of an existing level
        elif new_action.startswith("q"):
            lev = re.findall(r'\d+',new_action)[0]
            self.curr_price =  obs[col_locator["ASKp" + lev]]
            self.curr_level = int(new_action[-1])
            asks_cols = [col for col in self.orderbook.columns if col.startswith("ASKs") if (int(re.findall(r'\d+', col)[0]) <= self.curr_level)]
            self.askqueue_vol = self.askqueue_vol = np.sum(obs[[col_locator[col] for col in asks_cols]])
            self.curr_id = obs[col_locator["OrderID"]]

    def _update_weights(self, new_state, old_state, old_stateaction, actionset, reward, eligibility):

        # Compute TD(0) target
        delta = reward + self.gamma * np.dot(new_state, self.pcritic) - np.dot(old_state, self.pcritic)

        # Update critic parameter
        self.pcritic += self.pcritic_stepsize * delta * old_state

        # Update actor parameters
        self.pactor += self.pactor_stepsize * delta * eligibility
        self.pactor = np.clip(self.pactor, -10, 10)

        # Compute grad(log(pi))
        ex = np.zeros_like(old_stateaction)
        for a in actionset:
            ex += self._softmax(old_state, a) * self._stateaction_vec(old_state, a)

        # Update eligibility trace
        eligibility = self.lam * self.gamma * eligibility + ex
        return eligibility

    def _main_loop(self, initState, initAction, initMidprice, gen, freeze, sit_leave):
        """
        Perform main loop in self.learn()
        :param initState:
        :param initAction:
        :return:
        """
        old_state = initState
        old_action = initAction
        counter = 0
        col_locator = self.col_locator
        actionset = self.actionset
        cumreward = 0
        eligibility = np.zeros_like(self.pactor)


        while True:
            try:
                obs = self._get_new_obs(gen=gen)

            except StopIteration:
                print("End of period.")
                self.terminal = True
                r = self._get_reward(obs=obs)
                cumreward += r
                break

            old_stateaction = self._stateaction_vec(old_state, old_action) # numpy array of concatenated state and action array

            if (self.pactor is None) or (self.pcritic is None):
                self._initialize_weights(stateaction_vec=old_stateaction, state_vec=old_state) # we initialize weights here because there was
                                                                                               # no need to construct a full stateaction vector before

            counter += 1
            self.pactor_stepsize = 1/counter # step size
            self.pcritic_stepsize = 1/counter # step size
            self.curr_time = obs[col_locator["Timestamp"]]

            ### STEP ###
            step_start = time.time()
            new_state, reward = self._step(obs=obs)
            cumreward += reward

            ## Select new action
            if sit_leave:
                new_action = old_action
            else:
                plist = [self._softmax(new_state, act) for act in actionset]
                new_action = self._choose_action(obs=obs, new_state=new_state, plist=plist)

            # Update askqueue volume, current price, etc.
            self._housekeeping(new_action=new_action, initMidprice=initMidprice, obs=obs)

            # new state-action vector
            new_stateaction = self._stateaction_vec(new_state, new_action)


            ######################################################################## WEIGHT UPDATES
            upd_start = time.time()
            if not freeze:
                assert len(new_stateaction) == len(self.pactor), "new state-action vector is not the same length as theta"
                assert len(old_stateaction) == len(self.pactor), "old state-action vector is not the same length as theta"
                eligibility = self._update_weights(new_state=new_state,
                                                   old_state=old_state,
                                                   old_stateaction=old_stateaction,
                                                   actionset=actionset,
                                                   reward=reward,
                                                   eligibility=eligibility)


            old_state = new_state
            old_action = new_action

            # Check if we have no inventory left
            if self.rem_intory <=0:
                break


        return cumreward

    def _initialize_episode(self, obs):

        col_locator = self.col_locator
        initState = self._feat2tile(obs)
        initAction = self._choose_action(obs=obs)
        initMidprice = obs[col_locator["Midprice"]]

        # Weights
        # If training starts from anew, we initilialize the weights in the main loop

        # Curr_level, Curr_price, Curr_id
        self.curr_level = int(re.findall(r'\d+', str(initAction))[0])

        if initAction[0] == "i":
            self.curr_price = obs[col_locator["ASKp" + self.curr_level.__str__()]] - 0.01/initMidprice
        else:
            self.curr_price = obs[col_locator["ASKp" + self.curr_level.__str__()]]

        self.curr_id = obs[col_locator["OrderID"]]

        # initial ask volume in front of us
        asks_cols = [col for col in self.orderbook.columns if col.startswith("ASKs") if int(re.findall("\d+", col)[0]) <= self.curr_level]
        self.askqueue_vol = np.sum(obs[[col_locator[col] for col in asks_cols]])

        return initState, initAction, initMidprice

    def learn(self, nEpisodes, lobster_location, replay_path=None, freeze=False, sit_leave=False):
        """
        - learn - This function performs Sarsa(lambda) using linear function approximation
        Note: When an episode is loaded all prices are divided by the midprice of the starting time of the episode
        :return:
        """
        cep = 0 # episode counter
        cum_rewards = [] # save the cumulative rewards from each episode


        save_times = []

        for fname in os.listdir(replay_path):
            if fname.startswith("OTE_times_"):
                try:
                    replay_times = pickle.load(open(replay_path + fname, "rb"))
                    gen_time = (day_ix_tuple for day_ix_tuple in replay_times)
                    print("Replaying times...")
                    break
                except (FileNotFoundError, IOError):
                    print("No replay file found at ", replay_path)
                    print("Recording days and start indices...")
                    replay_path = None



        #### FOR EACH EPISODE ####
        while cep < nEpisodes:

            if replay_path is None:
                day = np.random.choice(self.dates) # Load orderbook of random day
            else:
                try:
                    day, stix = gen_time.__next__()
                except StopIteration:
                    print("All times in ", replay_path, " replayed.")
                    break



            self.load_data(day=day, lobster_location=lobster_location)

            # Choose random start time on that day
            if replay_path is None:
                start_ix = np.random.randint(0, len(self.orderbook.loc[self.orderbook.Timestamp < self.orderbook.Timestamp.iloc[-1] - self.ep_length]))
                save_times.append((day, start_ix)) # save if we want to replay the same training path lateron
            else:
                start_ix = stix

            start_time = self.orderbook["Timestamp"].iloc[start_ix]
            print("Trading Day: ", day, " Starting time: ", start_time)


            self.curr_time = start_time
            self.end_time = start_time + self.ep_length
            self.rem_intory = self.start_intory
            gen = self._ObsGenerator(starting_time=start_time)

            ######################################################################## INITIALIZATION
            try:
                obs = self._get_new_obs(gen=gen)
            except StopIteration:
                print("End of period.")
                break

            initState, initAction, initMidprice = self._initialize_episode(obs=obs)

            ######################################################################## MAIN LOOP
            cumreward = self._main_loop(initState=initState, initAction=initAction, initMidprice=initMidprice,
                                        gen=gen, freeze=freeze, sit_leave=sit_leave)

            cum_rewards.append(cumreward)
            cep += 1
            print("cumulative reward in episode %d: %0.3f" %(cep, cumreward))


        return cum_rewards, save_times
        #self._save_model(cum_rewards=cum_rewards, save_times=save_times, actor_stepsize=self.pactor_stepsize, critic_stepsize=self.pcritic_stepsize,
        #                 path="/Users/mh/Documents/CSML/Masterarbeit/Python/RL_haven/DEEP/remote/test/")

    def save_model(self, cum_rewards, save_times, actor_stepsize, critic_stepsize, replay_path):
        """
        Save model weights and
        :param cum_rewards:
        :return:
        """
        #path = "/home/mhoefl/OptimalTradeExecution/"


        pickle.dump(self.pcritic, open(checkpoint_path + "OTE_critic_params.p", "wb"))
        pickle.dump(self.pactor, open(checkpoint_path + "OTE_actor_params.p", "wb"))
        pickle.dump(actor_stepsize, open(checkpoint_path + "OTE_actor_stepsize.p", "wb"))
        pickle.dump(critic_stepsize, open(checkpoint_path + "OTE_critic_stepsize.p", "wb"))

        time = datetime.datetime.now()
        time = ''.join([str(time.year), str(time.month), str(time.day), str(time.hour), str(time.minute)])
        pickle.dump(cum_rewards, open(checkpoint_path + "OTE_cumrewards_%s.p" % time, "wb"))
        pickle.dump(save_times, open(replay_path + "OTE_times_%s" % time, "wb"))

if __name__ == "__main__":

    if len(sys.argv) == 1:
        ep_length = '3m'
        start_inventory=1000
        n_episodes = '5'
        lobster_location = "/Volumes/INTENSO/LOBSTER/pickledLobster/"
        checkpoint_path = "/Users/mh/Documents/CSML/Masterarbeit/Python/RL_haven/RAW/local/"
        replay_path = "/Users/mh/Documents/CSML/Masterarbeit/Python/RL_haven/RAW/local/"

    else:
        try:
            ep_length, start_inventory, n_episodes, lobster_location, checkpoint_path = sys.argv[1:]
        except IndexError:
            print("Usage: OTE.py\n<ep_length>\n<day>\n<n_episodes>\n<lobster_location>\n<checkpoint_path>")
            sys.exit(1)

    starttime = time.time()

    start_inventory = int(start_inventory)

    sarsa = SARSA_LINAPPROX(ep_length=ep_length, start_intory=start_inventory, checkpoint_path=checkpoint_path)

    crewards, save_times = sarsa.learn(nEpisodes=int(n_episodes), lobster_location=lobster_location, freeze=False,
                                       sit_leave=False)

    sarsa.save_model(cum_rewards=crewards, save_times=save_times,
                     actor_stepsize=sarsa.pactor_stepsize,
                     critic_stepsize=sarsa.pcritic_stepsize,
                     replay_path=replay_path)

    endtime = time.time()
    print("time elapsed: ", endtime - starttime)
