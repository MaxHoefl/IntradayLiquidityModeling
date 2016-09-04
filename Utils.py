from __future__ import division

import numpy as np
import pandas as pd

import pickle
import os
import sys
import collections
import h5py

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import TimeDistributed
from keras.layers.advanced_activations import LeakyReLU

from matplotlib import pyplot as plt


ticker = 'AAPL'
starttime = '34200000'
endtime = '57600000'
nlevels = '100'
lobster_location = "/Volumes/INTENSO/LOBSTER/pickledLobster/"

from LobsterData import LOBSTER

def save_lobsters(target_location, source_location, dates=None, extension='.pickle'):
    if dates is None:
        print("No dates specified.")
        sys.exit(1)

    for day in dates:
        print(day)
        fname = ticker + "_" + day + "_" + starttime + "_" + endtime + "_" + "lobsterclass_" + nlevels + ".pickle"
        if os.path.isfile(target_location + fname): print("Lobster class for day %s already pickled." %day)
        else:
            try:
                Lobster = LOBSTER(ticker=ticker, day=day, starttime=starttime, endtime=endtime,
                                  nlevels=nlevels, nlevels_keep=10, location=source_location, extension=extension)
                Lobster.load_book()
                Lobster.orderbook = aggregate_marketorders(Lobster.orderbook)
                Lobster.get_marketorders()
                Lobster.get_limitorder_submissions() # for lo_level
                Lobster.get_lasttrade_durations() # for LastTradeDuration

                with open(target_location + fname, "wb") as loc:
                    pickle.dump(Lobster, loc)
                print("Pickled Lobster class for day %s." %day)
            except (FileNotFoundError, OSError):
                print("File for day ", day, "not found.")
                continue
        print("Done.")

def plot_mean_net_order_flow(*Lobsters, save_results=False, price_moving_orders=False):

    loc = "/Users/mh/Documents/CSML/Masterarbeit/Python/"
    fig = plt.figure()
    num_lobster = len(Lobsters[0])
    ax_count = 0

    for Lobster in Lobsters[0]:

        if Lobster.orderbook is not None:
            orderbook = Lobster.orderbook
        else:
            Lobster.load_book()


        if Lobster.marketorders is not None:
            marketorders = Lobster.marketorders.loc[~Lobster.marketorders.duplicated()]
        else:
            Lobster.get_marketorders()


        if Lobster.limitorders is not None:
            limitorders = Lobster.limitorders
        else:
            Lobster.get_limitorder_submissions()

        # Get all marketorders that did not move the price (i.e. that did not consume the volume at best price)
        df = orderbook.ix[marketorders.index,["BIDs1","ASKs1"]].loc[:,["BIDs1","ASKs1"]]
        df = df.loc[~df.index.duplicated()]
        marketorders = pd.concat([marketorders,df],axis=1)

        ### Price maintaining, T-separated market orders
        # Keep only price preserving market orders
        mo_nomove = marketorders.loc[((marketorders.TradeDirection==-1) & (marketorders.Size<marketorders.ASKs1)) |
                                 ((marketorders.TradeDirection==1) & (marketorders.Size<marketorders.BIDs1))]
        # Keep only T separated marketorders where T=1[sec]
        mo_nomove = mo_nomove.loc[mo_nomove.IntertradeTimes > 1]

        ### Price changing, T-separated market orders
        mo_move = marketorders.loc[((marketorders.TradeDirection==-1) & (marketorders.Size>=marketorders.ASKs1)) |
                                 ((marketorders.TradeDirection==1) & (marketorders.Size>=marketorders.BIDs1))]
        # Keep only T separated marketorders where T=1[sec]
        mo_move = mo_move.loc[mo_move.IntertradeTimes > 1]

        # Compute the net order flow between t@i and t@i+tau, where t@i is the arrivaltime of the i-th marketorder
        # The net order flow is the cumulative difference between the limit orders that get added and cancelled
        taus = np.linspace(10**(-6),1,20)
        net_flow_dict = {}
        cancorders_bid = orderbook.loc[(orderbook.Type==2)|(orderbook.Type==3) & (orderbook.TradeDirection==1)]
        cancorders_ask = orderbook.loc[(orderbook.Type==2)|(orderbook.Type==3) & (orderbook.TradeDirection==-1)]

        limitorders_bid = limitorders.loc[limitorders.TradeDirection==1]
        limitorders_ask = limitorders.loc[limitorders.TradeDirection==-1]

        for tau in taus:
            print("Lobster on day %s, processing tau %f" %(Lobster.day, tau))
            net_bid_flow = []
            net_ask_flow = []
            if price_moving_orders:
                for cix in mo_move.index:
                    t_start = mo_move.ix[cix,"Time"]
                    tradedir = mo_move.ix[cix,"TradeDirection"]
                    if tradedir == 1:
                        net_flow = limitorders_ask.loc[(limitorders_ask.Time>t_start) & (limitorders_ask.Time<t_start+tau),"Size"].sum() - \
                                   cancorders_ask.loc[(cancorders_ask.Time>t_start) & (cancorders_ask.Time<t_start+tau),"Size"].sum()
                        net_bid_flow.append(net_flow)
                    else:
                        net_flow = limitorders_bid.loc[(limitorders_bid.Time>t_start) & (limitorders_bid.Time<t_start+tau),"Size"].sum() - \
                                   cancorders_bid.loc[(cancorders_bid.Time>t_start) & (cancorders_bid.Time<t_start+tau),"Size"].sum()
                        net_ask_flow.append(net_flow)
                net_flow_dict[tau] = (net_ask_flow, net_bid_flow)
            else:
                for cix in mo_nomove.index:
                    t_start = mo_nomove.ix[cix,"Time"]
                    tradedir = mo_nomove.ix[cix,"TradeDirection"]
                    if tradedir == 1:
                        net_flow = limitorders_ask.loc[(limitorders_ask.Time>t_start) & (limitorders_ask.Time<t_start+tau),"Size"].sum() - \
                                   cancorders_ask.loc[(cancorders_ask.Time>t_start) & (cancorders_ask.Time<t_start+tau),"Size"].sum()
                        net_bid_flow.append(net_flow)
                    else:
                        net_flow = limitorders_bid.loc[(limitorders_bid.Time>t_start) & (limitorders_bid.Time<t_start+tau),"Size"].sum() - \
                                   cancorders_bid.loc[(cancorders_bid.Time>t_start) & (cancorders_bid.Time<t_start+tau),"Size"].sum()
                        net_ask_flow.append(net_flow)
                net_flow_dict[tau] = (net_ask_flow, net_bid_flow)

        net_flow_dict = collections.OrderedDict(sorted(net_flow_dict.items()))

        if save_results:
            if price_moving_orders:
                fname = 'AAPL_' + Lobster.day + "_movingorders_netorderflow.p"
                pickle.dump(net_flow_dict,open(loc+fname,"wb"))
            else:
                fname = 'AAPL_' + Lobster.day + "_netorderflow.p"
                pickle.dump(net_flow_dict,open(loc+fname,"wb"))

        A = pd.DataFrame({})
        B = pd.DataFrame({})
        for ix,key in enumerate(net_flow_dict.keys()):
            A[key] = net_flow_dict[key][0]
            B[key] = net_flow_dict[key][1]

        ax_count += 1
        ax = fig.add_subplot(num_lobster,1,ax_count)
        A.mean(axis=0).plot(ax=ax)
        B.mean(axis=0).plot(ax=ax)
        if ax_count == 1:
            #plt.legend(["Mean net order flow - Ask","Mean net order flow - Bid"])
            plt.legend(["Mean net order flow - Ask","Mean net order flow - Bid"], loc='upper center', bbox_to_anchor=(0.5,1.05), ncol=2, fancybox=True)

    fig.suptitle("AAPL: Net order flow")
    if save_results:
        if price_moving_orders:
            fig.savefig(loc + "AAPL_movingorders_netorderflow.png")
        else:
            fig.savefig(loc + "AAPL_netorderflow.png")
    return fig

def perc(vals, k):
    return np.percentile(vals, q=k)

def _get_count(dates, lobster_location,ep_length='5m', n_eps=500):
    count = 0
    for day in dates:
        fname = ticker + "_" + day + "_" + starttime + "_" + endtime + "_" + "lobsterclass_" + nlevels + ".pickle"
        if os.path.isfile(lobster_location + fname):
            print("loading day %s" %day)

            # Load Lobster
            Lobster = pickle.load(open(lobster_location + fname, "rb"))

            # Pick random starttimes
            np.random.seed(seed=1992)
            rnd_rows = np.random.choice(np.arange(0,len(Lobster.orderbook.loc[Lobster.orderbook.Timestamp < Lobster.orderbook.Timestamp.iloc[-1] - pd.to_timedelta(ep_length)])),size=n_eps,replace=False)
            starttimes = Lobster.orderbook.Timestamp.iloc[rnd_rows]
            for st in starttimes:
                episode = Lobster.orderbook.loc[(Lobster.orderbook.Timestamp >= st) & (Lobster.orderbook.Timestamp <= st + pd.to_timedelta(ep_length))]
                count += len(episode)
    return count

def _get_askp_tiles(feature, lobster_location, percstep=5, ep_length='5m', n_eps=100):
    """
    Computes the average percentiles for ask prices on each level over episodes in each day.
    From given dates we load multiple episodes, compute the percentiles for the ask prices on each level
    and compute the average percentiles online.
    """
    # dates from which episodes are drawn
    dates = ["2014-06-0"+day.__str__() if len(day.__str__())==1 else "2014-06-"+day.__str__() for day in range(1,10)]

    # df_perc contains for each level an array of average percentiles over many episodes
    # if we compute tiles for ask prices then we are looking for the percentiles of ask prices at each level
    if feature == "ASKp":
        df_perc = pd.DataFrame(dict([("Lev"+k.__str__(),np.zeros((len(range(1,101,percstep)),)).tolist()) for k in range(1,11)]))
    else:
        df_perc = pd.DataFrame({feature : np.zeros((len(range(1,101,percstep)),)).tolist()})

    # Compute how many episode we gonna look at in total
    total_avgs = _get_count(dates=dates, lobster_location=lobster_location, ep_length=ep_length, n_eps=n_eps)

    for day in dates:
        fname = ticker + "_" + day + "_" + starttime + "_" + endtime + "_" + "lobsterclass_" + nlevels + ".pickle"
        if os.path.isfile(lobster_location + fname):
            print("loading day %s" %day)

            # Load Lobster
            Lobster = pickle.load(open(lobster_location + fname, "rb"))
            askp_cols = [col for col in Lobster.orderbook.columns if col.startswith("ASKp")]
            askp_cols.append("BIDp1")

            # Ensure all relevant columns are available in the orderbook
            if "Midprice" not in Lobster.orderbook.columns and (feature=="ASKp"):
                Lobster.orderbook.insert(1,"Midprice", Lobster.orderbook.BIDp1 + (Lobster.orderbook.ASKp1-Lobster.orderbook.BIDp1)/2)
            elif "LastTradeVolume" not in Lobster.orderbook.columns and (feature=="LastTradeVolume"):
                Lobster.get_lasttrade_volume()
            elif "LastTradeDuration" not in Lobster.orderbook.columns and (feature=="LastTradeDuration"):
                Lobster.get_lasttrade_durations()


            # Pick a constant seed for all days
            np.random.seed(seed=1992)

            # Pick random starting times for the episodes from the current orderbook
            rnd_rows = np.random.choice(np.arange(0,len(Lobster.orderbook.loc[Lobster.orderbook.Timestamp < Lobster.orderbook.Timestamp.iloc[-1] - pd.to_timedelta(ep_length)])),
                                        size=n_eps,replace=False)
            starttimes = Lobster.orderbook.Timestamp.iloc[rnd_rows]
            counter = 0

            for st in starttimes:
                counter +=1
                if counter % 100 == 0:
                    print(counter)

                # current episode
                episode = Lobster.orderbook.loc[(Lobster.orderbook.Timestamp >= st) & (Lobster.orderbook.Timestamp <= st + pd.to_timedelta(ep_length))]

                # If we are looking for percentiles of ask prices then divide the episode prices by the starting midprice
                if feature == "ASKp":
                    episode.loc[:,askp_cols] = episode.loc[:,askp_cols].div(episode.loc[:,"Midprice"], axis=0)


                # Get percentiles of the feature
                # if we compute tiles for ask prices then we compute the percentiles for every level separately
                if feature == "ASKp":
                    df_tmp = pd.DataFrame(dict([("Lev"+k.__str__(),[]) for k in range(1,11)]))
                    for ll in range(1,11):
                        df_tmp.loc[:,"Lev" + ll.__str__()] = np.asarray([perc(episode.loc[:,feature + ll.__str__()].values, k) for k in range(1,101,percstep)])

                else:
                    df_tmp = pd.DataFrame({})
                    df_tmp[feature] = np.asarray([perc(episode.loc[:,feature].values, k) for k in range(1,101,percstep)])

                # Update average percentile frame
                df_perc += 1/total_avgs * (df_tmp - df_perc)

    return df_perc

def _get_features4aft(day, features, lobster_location):
    """
    Build feature frame for accelerated time failure model
    :param features:
    :param lobster_location:
    :return:
    """
    assert day in ["2014-06-06", "2014-06-09"]

    ### Load Lobster class where we already have computed the order age
    print("Loading Lobster class...")
    fname = ticker + "_" + day + "_" + starttime + "_" + endtime + "_" + "lobsterclass_" + nlevels + ".pickle"
    Lobster = pickle.load(open(lobster_location + fname, "rb"))
    assert "order_age" in Lobster.orderbook.columns, "First call Lobster.get_order_age()."

    ### Average order age on bid and ask side
    Lobster.orderbook["avg_order_age"] = 0
    Lobster.orderbook.loc[Lobster.orderbook.TradeDirection==-1, "avg_order_age"] = \
        Lobster.orderbook.loc[Lobster.orderbook.TradeDirection==-1,["order_age"]].expanding(min_periods=1).mean()

    Lobster.orderbook.loc[Lobster.orderbook.TradeDirection==1, "avg_order_age"] = \
        Lobster.orderbook.loc[Lobster.orderbook.TradeDirection==1,["order_age"]].expanding(min_periods=1).mean()

    ### Total ask and bid volume at each time
    asks_cols = [col for col in Lobster.orderbook.columns if col.startswith("ASKs")]
    bids_cols = [col for col in Lobster.orderbook.columns if col.startswith("BIDs")]
    Lobster.orderbook["OrderImb"] = 0
    Lobster.orderbook.loc[:,"OrderImb"] = Lobster.orderbook.loc[:,asks_cols].sum(axis=1) - Lobster.orderbook.loc[:,bids_cols].sum(axis=1)


    ### Spread
    if "Spread" not in Lobster.orderbook.columns:
        Lobster.orderbook["Spread"] = Lobster.orderbook.ASKp1 - Lobster.orderbook.BIDp1

    ### Dummy indicating buy and sell marketorder
    Lobster.orderbook["MO_ask"] = 0
    Lobster.orderbook["MO_bid"] = 0
    Lobster.orderbook.loc[((Lobster.orderbook.Type==4)|(Lobster.orderbook.Type==5)) & (Lobster.orderbook.TradeDirection==1),"MO_bid"] = 1
    Lobster.orderbook.loc[((Lobster.orderbook.Type==4)|(Lobster.orderbook.Type==5)) & (Lobster.orderbook.TradeDirection==-1),"MO_ask"] = 1

    ### Threshold exceedance durations
    print("Computing TEDs...")
    TED_starts, TEDs = Lobster.get_TEDs(LiquMeasure='XLM')
    TED_starts = np.asarray(TED_starts)
    TEDs = np.asarray(TEDs)
    TED_starts = TED_starts[TEDs > 0]
    TEDs = TEDs[TEDs > 0]

    ### Construct feature frame
    df_feat = Lobster.orderbook
    df_feat = df_feat.loc[df_feat.Time.isin(TED_starts)]
    df_feat = df_feat.loc[~df_feat.Time.duplicated()]
    df_feat = df_feat.loc[:, features]

    assert df_feat.shape[0] == TEDs.shape[0], "feature frame has not as many rows as there are teds."
    df_feat["TED"] = TEDs
    df_feat["TED_starts"] = TED_starts

    df_feat["LastTED"] = df_feat.TED_starts - (df_feat.TED_starts.shift(1) + df_feat.TED.shift(1))
    df_feat["AvgTED_m5"] = df_feat.loc[:,"TED"].rolling(window=5).mean() # average over last 5 TEDs
    df_feat["AvgTED_m5"].iloc[:4] = df_feat["TED"].iloc[:4].expanding(min_periods=1).mean()

    df_feat = df_feat.iloc[1:]
    del df_feat["TED_starts"]

    return df_feat

def concat_dataframes(dates, lob_loc, model_type='LSTM', binary_encoding=False):
    print("model type: %s" % model_type)
    data_X = None
    data_Y = None
    lob_loc = lob_loc
    for day in dates:
        fname = ticker + "_" + day + "_" + starttime + "_" + endtime + "_" + "lobsterclass_" + nlevels + ".pickle"
        if os.path.exists(lob_loc + fname):
            X,y = _get_data(day, "pandas", lob_loc)
        else:
            print("%s does not exist!" %(lob_loc + fname))
            continue

        if data_X is None:
            data_X = X
            data_Y = y
        else:
            data_X = pd.concat([data_X, X])
            data_Y = pd.concat([data_Y, y])

    if binary_encoding:
        data_X = np.sign(data_X - data_X.shift(1))
        data_X = data_X.iloc[1:]
        data_Y = data_Y.iloc[1:]
    else:
        # Standardize variables
        data_X = data_X - data_X.mean(axis=0)
        std = (data_X ** 2).sum(axis=0)
        data_X /= std

    if isinstance(data_X, pd.DataFrame):
        data_X, data_Y =  data_X.values, data_Y.values

    return data_X, data_Y

def _get_data(day, datatype, lobster_location):

    print(day)
    # features
    feat_cols = ["Type", "LastTradeDuration","LastTradeVolume"]
    ask_cols = ["ASKp" + lev.__str__() for lev in range(1,11)]
    ask_cols.extend(["ASKs" + lev.__str__() for lev in range(1,11)])
    bid_cols = ["BIDp" + lev.__str__() for lev in range(1,11)]
    bid_cols.extend(["BIDs" + lev.__str__() for lev in range(1,11)])
    feat_cols.extend(ask_cols)
    feat_cols.extend(bid_cols)

    # orderbook
    fname = ticker + "_" + day + "_" + starttime + "_" + endtime + "_" + "lobsterclass_" + nlevels + ".pickle"
    Lobster = pickle.load(open(lobster_location + fname, "rb"))
    orderbook = Lobster.orderbook
    orderbook = orderbook.loc[:, feat_cols] # only keep selected features

    # construct data frame
    df_feat = orderbook
    df_feat["m_order"] = 0
    df_feat.loc[(df_feat.Type==4) | (df_feat.Type==5), "m_order"] = 1
    df_feat.loc[:,"m_order"] = df_feat.loc[:,"m_order"].shift(-1)
    df_feat = df_feat.iloc[:-1]

    # features and responses
    X = df_feat.loc[:,[col for col in df_feat.columns if (col != "m_order")]]
    y = df_feat.loc[:,["m_order"]]
    del X["Type"]
    del X["LastTradeVolume"]
    del X["LastTradeDuration"]

    if datatype == "pandas":
        return X, y
    elif datatype == "numpy":
        return X.values, y.values
    else:
        print("datatype must be either 'pandas' or 'numpy'.")
        return 0,0

def _get_rolling_windows(X, Y=None, size=10):
    assert len(X.shape) == 2, "Pass 2D array into get_rolling_windows"
    nobs = X.shape[0]
    nfeat = X.shape[1]
    X_rw = np.zeros((nobs - size + 1, size, nfeat))

    for i in range(size-1, nobs):
        if i == nobs - 1:
            X_rw[i-size+1, :, :] = X[(i-size+1):,:]
        else:
            X_rw[i-size+1, :, :] = X[(i-size+1):(i+1),:]

    if Y is None:
        return X_rw
    else:
        return X_rw, Y[size-1:,]

def get_summary_stats(dates, stats, stats_path):
    d = dict([(stat, None) for stat in stats])
    num_days = len(dates)
    counter = 0

    for day in dates:
        try:
            fname = ticker + "_" + day + "_" + starttime + "_" + endtime + "_" + "lobsterclass_" + nlevels + ".pickle"
            with open(lobster_location + fname, "rb") as file:
                Lobster = pickle.load(file)
                print(day)

                counter += 1

                limitorders = Lobster.limitorders
                marketorders = Lobster.marketorders
                orderbook = Lobster.orderbook
                orderbook = orderbook.reset_index(drop=True)
                #orderbook.loc[:,"Timestamp"] = pd.to_datetime(orderbook.Timestamp)
                cancellations = orderbook.loc[(orderbook.Type == 2)|(orderbook.Type == 3)]

                # Get for each executed limit order its indices
                tsorted_orderlife_exec = Lobster.get_orderlives(orderbook=orderbook, mode='none', legit_type=[1,4,5])
                tsorted_orderlife_canc = Lobster.get_orderlives(orderbook=orderbook, mode='none', legit_type=[1,2,3])
                tsorted_orderlife_aggr = Lobster.get_orderlives(orderbook=orderbook, mode='aggr', legit_type=[1,2,3])

                try:
                    LO_ages_exec = pickle.load(open(stats_path + "LO_ages_exec_%s.p" % day, "rb"))
                except FileNotFoundError:
                    tsorted_orderlife_exec = Lobster.get_orderlives(orderbook=orderbook, mode='none', legit_type=[1,4,5])
                    LO_ages_exec = [(val[-1]-val[0]).value for val in tsorted_orderlife_exec.values()]
                    pickle.dump(LO_ages_exec, open(stats_path + "LO_ages_exec_%s.p" % day, "wb"))

                try:
                    LO_ages_canc = pickle.load(open(stats_path + "LO_ages_canc_%s.p" % day, "rb"))
                except FileNotFoundError:
                    LO_ages_canc = [(val[-1]-val[0]).value for val in tsorted_orderlife_canc.values()]
                    pickle.dump(LO_ages_canc, open(stats_path + "LO_ages_canc_%s.p" % day, "wb"))

                try:
                    VLO_sizes = pickle.load(open(stats_path + "VLO_sizes_%s.p" % day, "rb"))
                except FileNotFoundError:
                    VLO_sizes = [marketorders.loc[orderbook.Timestamp == tstamp[0], "Size"].values for tstamp in tsorted_orderlife_exec.values()]
                    pickle.dump(VLO_sizes, open(stats_path + "VLO_sizes_%s.p" % day, "wb"))


                try:
                    ALO_ages_canc = pickle.load(open(stats_path + "ALO_ages_canc_%s.p" % day, "rb"))
                except FileNotFoundError:
                    ALO_ages_canc = [(val[-1]-val[0]).value for val in tsorted_orderlife_canc.values()]
                    pickle.dump(ALO_ages_canc, open(stats_path + "ALO_ages_canc_%s.p" % day, "wb"))

                VLO_ages_exec = LO_ages_exec





                """
                LO_ages_exec = []
                LO_ages_canc = []
                VLO_ages_exec = []
                for event_indices in tsorted_orderlife.values():
                    birth = event_indices[0]
                    death = event_indices[-1]
                    age = (orderbook.ix[death].Timestamp - orderbook.ix[birth].Timestamp).value

                    if (orderbook.ix[death].Type == 4) | (orderbook.ix[death].Type == 5):
                        vweighted_age = orderbook.ix[birth].Size * age

                        LO_ages_exec.append(age) # life time of limit orders that get executed
                        VLO_ages_exec.append(vweighted_age)

                    elif (orderbook.ix[death].Type == 2) | (orderbook.ix[death].Type == 3):
                        LO_ages_canc.append(age)


                print("     Orderlife of aggressive limitorders...")
                tsorted_orderlife_aggr = Lobster.get_orderlives(orderbook=orderbook, mode='aggr')

                ALO_ages_canc = []
                for event_indices in tsorted_orderlife_aggr.values():
                    birth = event_indices[0]
                    death = event_indices[-1]
                    age = (orderbook.ix[death].Timestamp - orderbook.ix[birth].Timestamp).value
                    if (orderbook.ix[death].Type == 2) | (orderbook.ix[death].Type == 3):
                        ALO_ages_canc.append(age)

                """
                LO_ages_exec = np.array(LO_ages_exec) / 1000
                LO_ages_canc = np.array(LO_ages_canc) / 1000
                VLO_ages_exec = np.array(VLO_ages_exec) / 1000
                VLO_sizes = np.array(VLO_sizes)
                print(len(VLO_sizes), len(VLO_ages_exec))
                assert len(VLO_sizes) == len(VLO_ages_exec)
                print(VLO_sizes)
                VLO_ages_exec = VLO_ages_exec * VLO_sizes
                ALO_ages_canc = np.array(ALO_ages_canc) / 1000

                print("     Updating statistics...")
                for stat in stats:
                    if stat == "AvgNumLO":
                        if d[stat] is None:
                            d[stat] = len(limitorders)
                        else:
                            d[stat] = d[stat] + 1/counter * (len(limitorders) - d[stat])

                    elif stat == "AvgSZ":
                        if d[stat] is None:
                            d[stat] = limitorders.Size.mean()
                        else:
                            d[stat] = d[stat] + 1/counter * (limitorders.Size.mean() - d[stat])

                    elif stat == "AvgNumALO":
                        if d[stat] is None:
                            d[stat] = len(tsorted_orderlife_aggr)
                        else:
                            d[stat] = d[stat] + 1/counter * (len(tsorted_orderlife_aggr) - d[stat])

                    elif stat == "AvgNumALOP":
                        if d[stat] is None:
                            d[stat] = len(tsorted_orderlife_aggr) / len(limitorders)
                        else:
                            d[stat] = d[stat] + 1/counter * (len(tsorted_orderlife_aggr) / len(limitorders) - d[stat])

                    elif stat == "AvgNumExe":
                        if d[stat] is None:
                            d[stat] = len(marketorders)
                        else:
                            d[stat] = d[stat] + 1/counter * (len(marketorders) - d[stat])

                    elif stat == "AvgETime":
                        if d[stat] is None:
                            d[stat] = np.mean(LO_ages_exec)
                        else:
                            d[stat] = d[stat] + 1/counter * (np.mean(LO_ages_exec) - d[stat])

                    elif stat == "VwAvgETime":

                        if d[stat] is None:
                            d[stat] = np.mean(VLO_ages_exec)
                        else:
                            #print(np.mean(VLO_ages_exec))
                            #print(VLO_ages_exec)
                            d[stat] = d[stat] + 1/counter * (np.mean(VLO_ages_exec) - d[stat])
                        #print(d[stat])

                    elif stat == "AvgNumCanc":
                        if d[stat] is None:
                            d[stat] = len(cancellations)
                        else:
                            d[stat] = d[stat] + 1/counter * (len(cancellations) - d[stat])

                    elif stat == "AvgNumCancP":
                        if d[stat] is None:
                            d[stat] = len(cancellations) / len(orderbook)
                        else:
                            d[stat] = d[stat] + 1/counter * (len(cancellations) / len(orderbook) - d[stat])

                    elif stat == "AvgCTim":
                        if d[stat] is None:
                            d[stat] = np.mean(LO_ages_canc)
                        else:
                            d[stat] = d[stat] + 1/counter * (np.mean(LO_ages_canc) - d[stat])

                    elif stat == "NumACan":
                        if d[stat] is None:
                            d[stat] = len(ALO_ages_canc)
                        else:
                            d[stat] = d[stat] + 1/counter * (len(ALO_ages_canc) - d[stat])

                    elif stat == "AvgACTime":
                        if d[stat] is None:
                            d[stat] = np.mean(ALO_ages_canc)
                        else:
                            d[stat] = d[stat] + 1/counter * (np.mean(ALO_ages_canc) - d[stat])


        except (FileNotFoundError, IOError):
            print("File for day ", day, " not found.")
    return d

def load_lstm(LSTM_PATH):
    """
    Load LSTM
    :return:
    """
    nb_tsteps = 10
    nb_feats = 40
    model = Sequential()
    model.add(TimeDistributed(Dense(120, activation = 'linear'), input_shape=(nb_tsteps, nb_feats)))
    model.add(LeakyReLU(alpha=.01))
    model.add(Dropout(0.2))
    model.add(LSTM(250, return_sequences=False, consume_less='gpu'))
    model.add(Dense(100, activation='linear'))
    model.add(LeakyReLU(alpha=.01))
    model.add(Dropout(0.2))
    model.add(Dense(60, activation='linear'))
    model.add(LeakyReLU(alpha=.01))
    model.add(Dropout(0.2))
    model.add(Dense(30, activation='linear'))
    model.add(LeakyReLU(alpha=.01))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='linear'))
    model.add(LeakyReLU(alpha=.01))
    model.add(Dropout(0.1))
    model.add(Dense(3, activation='linear'))
    model.add(LeakyReLU(alpha=.01))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    #path = "/Users/mh/Documents/CSML/Masterarbeit/Python/LSTM_v2/m03/"
    with h5py.File(LSTM_PATH + 'modelweights_LSTM_v2.h5','r+') as f:
        timedistributed_1 = model.layers[0]
        g_td1 = f["timedistributed_1"]
        td1_weights = [g_td1["dense_1_W"], g_td1["dense_1_b"]]
        timedistributed_1.set_weights(td1_weights)

        lstm_1 = model.layers[3]
        g_lstm1 = f["lstm_1"]
        lstm1_weights = [g_lstm1["lstm_1_W"], g_lstm1["lstm_1_U"], g_lstm1["lstm_1_b"]]
        lstm_1.set_weights(lstm1_weights)

        dense_2 = model.layers[4]
        g_dense2 = f["dense_2"]
        dense2_weights = [g_dense2["dense_2_W"], g_dense2["dense_2_b"]]
        dense_2.set_weights(dense2_weights)

        dense_3 = model.layers[7]
        g_dense3 = f["dense_3"]
        dense3_weights = [g_dense3["dense_3_W"], g_dense3["dense_3_b"]]
        dense_3.set_weights(dense3_weights)

        dense_4 = model.layers[10]
        g_dense4 = f["dense_4"]
        dense4_weights = [g_dense4["dense_4_W"], g_dense4["dense_4_b"]]
        dense_4.set_weights(dense4_weights)

        dense_5 = model.layers[13]
        g_dense5 = f["dense_5"]
        dense5_weights = [g_dense5["dense_5_W"], g_dense5["dense_5_b"]]
        dense_5.set_weights(dense5_weights)

        dense_6 = model.layers[16]
        g_dense6 = f["dense_6"]
        dense6_weights = [g_dense6["dense_6_W"], g_dense6["dense_6_b"]]
        dense_6.set_weights(dense6_weights)

        dense_7 = model.layers[18]
        g_dense7 = f["dense_7"]
        dense7_weights = [g_dense7["dense_7_W"], g_dense7["dense_7_b"]]
        dense_7.set_weights(dense7_weights)

    return model

def aggregate_marketorders(orderbook):
    """
    This method makes sure that we have the marketorders filling multiple limitorders summarized as one market event
        - Load LOBSTER instances from dates
        - Sum simultaneous marketorders with respect to size
    """

    # We are facing the problem that there are sometime also simultaneous non-marketorder events
    # If we just blindly aggregate over same timestamps we catch them accidantelly as well
    # The column groupby_flag is to ensure only marketorders at same timestamp have the same flag
    orderbook["groupby_flag"] = range(len(orderbook))

    # Construct dictionary that holds for each timestamp of simultaneous marketorder events a different flag
    mo_groups = orderbook.loc[(orderbook.Type==4) | (orderbook.Type==5)].groupby("Timestamp").agg({"groupby_flag":"first"})
    mo_dict = mo_groups.to_dict(orient='index')
    print("      Aggregating...")
    orderbook.loc[(orderbook.Type==4)|(orderbook.Type==5), "groupby_flag"] = \
             orderbook.loc[(orderbook.Type==4)|(orderbook.Type==5), "Timestamp"].apply(lambda tstamp: mo_dict[tstamp]["groupby_flag"])

    # Now we can group by groupby_flag
    agg_dict = {"Time":"first",
                "Timestamp":"first",
                "Day":"first",
                "Month":"first",
                "Year":"first",
                "Hour":"first",
                "Minute":"first",
                "Second":"first",
                "Millisecond":"first",
                "Microsecond":"first",
                "Nanosecond":"first",
                "Price": "first",
                "Type":"first",
                "OrderID": "first",
                "Size":"sum",
                "TradeDirection":"first"
                }

    for col in orderbook.columns:
        if col.startswith("ASK") | col.startswith("BID"):
            agg_dict[col] = "first"

    agg_set = set(agg_dict.keys())
    feat_set = set(orderbook.columns)
    feat_set.remove("groupby_flag")

    if not len(agg_set.difference(feat_set)) == 0:
        print("Features ", feat_set.difference(agg_set), " are not accounted for in agg_dict!")
        raise ValueError

    orderbook = orderbook.groupby("groupby_flag").agg(agg_dict)
    return orderbook

def precompute_lstm_predictions(dates, LSTM_PATH, TARGET_PATH, SOURCE_PATH):
    """
    Repeat for all days in dates:
        - Load pickled LOBSTER instance from that day
        - Compute the LSTM predictions for that day
        - Augment orderbook by bidask_state column holding all the predictions (thresholded at 0.4125)

    :param dates: days to select episodes from
    :param LSTM_PATH: path to lstm weight (h5py file) and model json
    :param LOSBTER_PATH: path to pickled LOBSTER instances
    :return: None
    """
    model = load_lstm(LSTM_PATH)
    batch_size = 139
    nb_tsteps = 10
    for day in dates:

        fname = ticker + "_" + day + "_" + starttime + "_" + endtime + "_" + "lobsterclass_" + nlevels + ".pickle"
        try:
            print(day)
            Lobster = pickle.load(open(SOURCE_PATH + fname, "rb"))
            orderbook = Lobster.orderbook
            orderbook = orderbook.iloc[:(len(orderbook)-(len(orderbook) % batch_size))]
            orderbook["bidask_state"] = 0
            lstm_features = [col for col in orderbook.columns if (col.startswith("ASK") or col.startswith("BID")) if (col!="ASKq")]
            X = orderbook.loc[:, lstm_features]
            X = np.sign((X - X.shift(1)).iloc[1:].values)

            # Construct rolling windows
            print("     Get rolling windows...")
            X_rw = _get_rolling_windows(X=X, size=10)

            print("     Predicting marketorders...")
            y_pred = model.predict(x=X_rw, batch_size=batch_size, verbose=1)
            orderbook = orderbook.iloc[1:]
            orderbook["bidask_state"].iloc[nb_tsteps-1:] = y_pred.flatten()
            orderbook = orderbook.iloc[nb_tsteps-1:]
            orderbook.loc[orderbook.bidask_state >= 0.4125, "bidask_state"] = 1
            orderbook.loc[orderbook.bidask_state < 0.4125, "bidask_state"] = 0

            Lobster.orderbook = orderbook
            pickle.dump(Lobster, open(TARGET_PATH + fname, "wb"))

        except (FileNotFoundError, IOError):
            print("     pickled Lobster instance not found at ", SOURCE_PATH)



if __name__ == "__main__":
    ticker = 'AAPL'
    starttime = '34200000'
    endtime = '57600000'
    nlevels = '100'

    target_location = '/Volumes/INTENSO/LOBSTER/pickledLobster/v02/'
    source_location = '/Volumes/INTENSO/LOBSTER/pickledLobster/'
    LSTM_PATH = "/Users/mh/Documents/CSML/Masterarbeit/Python/LSTM_v2/m04/"
    dates = ["2014-07-0" + cb.__str__() if cb < 10 else "2014-07-" + cb.__str__() for cb in range(24,32)]

    #save_lobsters(target_location=target_location, source_location=source_location, dates=dates, extension=".csv")
    precompute_lstm_predictions(dates=dates, LSTM_PATH=LSTM_PATH, TARGET_PATH=target_location, SOURCE_PATH=source_location)