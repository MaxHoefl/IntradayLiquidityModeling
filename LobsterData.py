from __future__ import division
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pickle
import copy as cp
import operator
import re
import sys
from statsmodels.distributions.empirical_distribution import ECDF



class LOBSTER():

    def __init__(self,ticker, day, starttime, endtime, nlevels, nlevels_keep, location, extension):

        if not (isinstance(ticker, str) & isinstance(day, str) & isinstance(starttime, str) & isinstance(endtime, str) & isinstance(nlevels, str)):
            print("Pass all required arguments as strings.")
            raise IOError

        lob_file_name = '_'.join([ticker, day, starttime, endtime, "orderbook", nlevels])
        msg_file_name = '_'.join([ticker, day, starttime, endtime, "message", nlevels])

        self.lob_path = ''.join([location, lob_file_name, extension])
        self.msg_path = ''.join([location, msg_file_name, extension])

        self.ticker = ticker
        self.day = day
        self.extension = extension
        self.nlevels = nlevels
        self.nlevels_keep = nlevels_keep
        self.date = day

        self.orderbook = None
        self.marketorders = None
        self.limitorders = None

    def get_orderlives(self, orderbook, mode, legit_type):
        """
        This is a helper method for plot_orderflow.
        :return: Return a list of tuples of the form
                    (seconds, [index of birth of order ID, index of event for order ID,..., index of event for order ID,
                        index of death of order ID])
                For example
                    (34669378, [7374,      7376])
                      ^          ^           ^
                    seconds    indexBirth   indexDeath
                An order dies either by cancellation or by execution
                Events are partial fillings
        """
        assert isinstance(legit_type, list), "Usage: legit_type must be list of order types to be considered."
        #assert legit_type.issubset({1,2,3,4,5}), "Usage: legit_type can only contain 1 through 5."
        assert mode in ['exec', 'aggr', 'none'], "Usage: mode is one of ['exec', 'aggr', 'none']."

        assert not any(orderbook.index.duplicated()), \
            print("found duplicate indices in orderbook.\nPlease do orderbook.reset_index(drop=True) before passing it again.")

        assert self.limitorders is not None, "First call: LOBSTER.get_limitorder_submissions()."
        limitorders = self.limitorders

        # Look only for order IDs belonging to aggressive limit orders (posted inside the spread)
        if mode == 'aggr':
            ALO_ask = limitorders.loc[limitorders.TradeDirection==-1]
            ALO_ask.loc[:,"ASKp1"] = ALO_ask.ASKp1.shift(1)
            ALO_ask = ALO_ask.iloc[1:]

            ALO_bid = limitorders.loc[limitorders.TradeDirection==1]
            ALO_bid.loc[:,"BIDp1"] = ALO_bid.BIDp1.shift(1)
            ALO_bid = ALO_bid.iloc[1:]

            ALO_ask = ALO_ask.loc[(ALO_ask.Price < ALO_ask.ASKp1),["Timestamp"]]

            ALO_bid = ALO_bid.loc[(ALO_bid.Price > ALO_bid.BIDp1),["Timestamp"]]
            ALO = pd.concat([ALO_ask, ALO_bid])

        tmp = cp.deepcopy(orderbook)
        del_orderIDs = tmp.loc[~tmp.Type.isin(legit_type), "OrderID"].unique()
        tmp = tmp.loc[~tmp.OrderID.isin(del_orderIDs)]
        tmp = tmp.set_index("Timestamp", drop=False)

        orderdict = tmp.groupby("OrderID").groups

        orderdict.pop(0, orderdict)
        drop_keys = []

        for key in orderdict.keys():
            if len(orderdict[key]) == 1:
                drop_keys.append(key)
                continue

            #elif set(orderbook.loc[orderbook.OrderID == key, ["Type"]]).issubset(legit_type):
            #    drop_keys.append(key)
            #    continue

            elif mode == 'aggr':
                if not orderdict[key][0] in ALO.Timestamp:
                    drop_keys.append(key)





        for k in drop_keys:
            orderdict.pop(k, None)

        #timesorted_orderlist = sorted(orderdict.items(), key=operator.itemgetter(1))
        return orderdict

    def get_intertrade_times(self,timedelta, freq):
        """
        In 'get_marketorders' (return type 'pd.DataFrame') we have a column 'IntertradeTimes' which is the time
        difference between two marketorders as timedelta. This method computes the number of seconds or nanoseconds
        that this timedelta represents.
        :param timedelta: The timedelta in a specific line of marketorders.IntertradeTimes
        :param freq: 'seconds' if the result should be in seconds, 'nanoseconds' if the result should be in nanoseconds
        """
        l = ['seconds', 'nanoseconds']
        assert freq in l
        try:
            timedelta = timedelta.components
            if freq == 'seconds':
                return (timedelta.minutes * 60 +
                       timedelta.seconds +
                       timedelta.milliseconds/1000 +
                       timedelta.microseconds/(1000 * 1000) +
                       timedelta.nanoseconds/(1000 * 1000 * 1000))

            elif freq == 'nanoseconds':
                return (timedelta.minutes * 60 * 1000 * 1000 * 1000 +
                       timedelta.seconds * 1000 * 1000 * 1000 +
                       timedelta.milliseconds * 1000 * 1000 +
                       timedelta.microseconds * 1000 +
                       timedelta.nanoseconds)
        except AttributeError:
            return 0

    def load_book(self):
        """
        This method loads the orderbook and messagebook from LOBSTER data (either preprocessed .pickle file or
        unprocessed .csv file) and inserts time columns.
        """
        if self.orderbook is not None:
            return None

        ### LOAD ORDERBOOK AND SET COLUMN NAMES
        print("     Loading book...")
        if self.extension == '.pickle':
            # If a pickle is loaded we assume that the file has already been modified. In particular we assume that
            # the pickle file is a pandas dataframe with set column headers and there are only 'nlevels_keep' levels.
            with open(self.lob_path, "rb") as lobpath:
                df_lob = pickle.load(lobpath)
                self.orderbook = df_lob

        elif self.extension == '.csv':
            # Here we assume that the .csv file has not been messed with, i.e. it is exactly how it comes out of LOBSTER
            # Specifically, the column headers aren't set, and there are 'nlevels' levels from whic
            orderbook = pd.read_csv(self.lob_path, header=-1)
            # There shold be at least 4 column dedicated to bid/ask price/vol level 1.
            assert int(self.nlevels) >= 1
            columns = ["ASKp1" , "ASKs1" , "BIDp1",  "BIDs1"]
            # If there are more than 1 level extend column names accordingly
            if int(self.nlevels) > 1:
                counter = 1
                for i in range(4*int(self.nlevels)-4):
                    if i % 4 == 0:
                        counter += 1
                        columns.extend(["ASKp" + str(counter), "ASKs" + str(counter), "BIDp" + str(counter), "BIDs" + str(counter)])
                    else:
                        continue
            # Set column names
            orderbook.columns = columns
            # Remove bid/ask columns where the level > nlevels_keep
            bid_ask_cols = [col for col in orderbook.columns if ((col.startswith("ASK")) | (col.startswith("BID")))]
            rel_cols = [col for col in bid_ask_cols if int(re.findall(r'\d+$', col)[0]) <= self.nlevels_keep]
            orderbook = orderbook.loc[:, rel_cols]

            ### LOAD MESSAGES AND SET COLUMN NAMES
            messages = pd.read_csv(self.msg_path, header=-1)
            messages.columns = ["Time" , "Type" , "OrderID" , "Size" , "Price" , "TradeDirection"]
            messages.Price = messages.Price/10000
            self.messages = messages
            self.raw_orderbook = orderbook
            ### CREATE AGGREGATE ORDERBOOK (WITH TRADES)
            self.orderbook = pd.concat([messages, orderbook], axis=1)
            try:
                self.orderbook.insert(1,"Hour",np.floor(self.orderbook.Time / (60*60)))
                self.orderbook.loc[:, "Hour"] = self.orderbook.Hour.astype(int)

                self.orderbook.insert(2,"Minute", (self.orderbook.Time / 60 - self.orderbook.Hour * 60))
                self.orderbook.loc[:, "Minute"] = self.orderbook.Minute.astype(int)

                self.orderbook.insert(3,"Second", (self.orderbook.Time - self.orderbook.Hour * 3600 - self.orderbook.Minute * 60))
                self.orderbook.loc[:, "Second"] = self.orderbook.Second.astype(int)

                self.orderbook.insert(4,"Millisecond", ((self.orderbook.Time - self.orderbook.Hour * 3600 - self.orderbook.Minute * 60 - self.orderbook.Second)*1000))
                self.orderbook.loc[:, "Millisecond"] = self.orderbook.Millisecond.astype(int)

                self.orderbook.insert(5,"Microsecond", ((self.orderbook.Time - self.orderbook.Hour * 3600 - self.orderbook.Minute * 60 - self.orderbook.Second - self.orderbook.Millisecond/1000)*1000000))
                self.orderbook.loc[:, "Microsecond"] = self.orderbook.Microsecond.astype(int)

                self.orderbook.insert(6,"Nanosecond", ((self.orderbook.Time - self.orderbook.Hour * 3600 - self.orderbook.Minute * 60 - self.orderbook.Second - self.orderbook.Millisecond/1000 - self.orderbook.Microsecond/1000000)*1000000000))
                self.orderbook.loc[:, "Nanosecond"] = self.orderbook.Nanosecond.astype(float)

                self.orderbook = self.orderbook[~((self.orderbook.Hour==9) & (self.orderbook.Minute<45) | ((self.orderbook.Hour==15) & (self.orderbook.Minute>45)))]
                # divide all prices by 10000 (to get dollar values)
                bid_ask_price_cols = [col for col in self.orderbook.columns if ((col.startswith("BIDp")) | (col.startswith("ASKp")))]
                self.orderbook.loc[:, bid_ask_price_cols] /= 10000

            except AttributeError as ae:
                print("Caught some attribute error when trying to insert time columns in orderbook")
                raise ae

        else:
            print('Use either .csv or .pickle as extension.')
            raise IOError

        date  = pd.to_datetime(self.date, format = "%Y-%m-%d")
        self.orderbook.insert(1, "Year", date.year)
        self.orderbook.insert(1, "Month", date.month)
        self.orderbook.insert(1, "Day", date.day)
        self.orderbook.insert(1,"Timestamp", pd.to_datetime(self.orderbook.loc[:, ["Year",
                                                                                     "Month",
                                                                                     "Day",
                                                                                     "Hour",
                                                                                     "Minute",
                                                                                     "Second",
                                                                                     "Millisecond",
                                                                                     "Microsecond",
                                                                                     "Nanosecond"]]))

        self.orderbook = self.orderbook.reset_index(drop=True)

    def restrict_time(self, startTime, endTime, df=None):
        """
        :param startTime: starting time of the frame we want to get
        :param endTime: ending time of the frame we want to get
        :param df: the frame in which we truncate the time to [startTime, endTime]
        :return: time-truncated DataFrame
        """
        assert isinstance(startTime,str)
        assert isinstance(endTime,str)

        if df is None:
            df = self.orderbook

        tstart = pd.to_datetime(startTime, format="%H:%M:%S")
        tend = pd.to_datetime(endTime, format="%H:%M:%S")
        df = df.loc[((df.Hour==tstart.hour) & (df.Hour>=tstart.minute)) |
                    ((df.Hour>tstart.hour) & (df.Hour<tend.hour)) |
                    ((df.Hour==tend.hour) & (df.Minute<=tend.minute))]

        return df

    def plot_orderlife(self,hour,minute=None, second=None):
        """
        Plot each order on the y-axis ascending with time and the birth and death time of the order on the x-axis
        Cancellations and deletions are marked red
        (Partial) fillings are marked yellow
        :param hour, minute, second: The time at which the orderflow should be plotted
        """

        if minute is None:
            lob = self.orderbook.loc[self.orderbook.Hour == hour]
            #t_min = hour * 3600
            #t_max = (hour+1) * 3600
        elif (second is None) & (minute is not None):
            lob = self.orderbook.loc[(self.orderbook.Hour == hour) & (self.orderbook.Minute == minute)]
            #t_min = hour * 3600 + minute * 60
            #t_max = hour * 3600 + (minute+1) * 60
        else:
            lob = self.orderbook.loc[(self.orderbook.Hour == hour) & (self.orderbook.Minute == minute) & (self.orderbook.Second == second)]
            #t_min = t_min = hour * 3600 + minute * 60 + second
            #t_max = t_min = hour * 3600 + minute * 60 + second + 1
        timesorted_orderlist = self.get_sorted_orderlist(lob)
        if len(timesorted_orderlist) == 0:
            print("Nothing happened in that time frame, except some deaths of opening auction orders.")
            return None
        fig = plt.figure(figsize=(20,15))
        ax = fig.add_subplot(111)

        t_min = lob.ix[timesorted_orderlist[0][0]].Time.iloc[0]
        t_max = lob.ix[timesorted_orderlist[-1][0]].Time.iloc[-1]
        #ax.set_xlim(t_min, t_max)

        counter = 0
        for item in timesorted_orderlist:
            counter += 1
            #if counter > 1000:
            #    break
            for pp in item[1]:
                # straight cancellations (no partial fillings)
                if (len(item[1])==2) & ((lob.ix[item[1]].Type.iloc[-1] == 2) | ((lob.ix[item[1]].Type.iloc[-1] == 3))):
                    ax.plot([tt for tt in lob.ix[item[1]].Time.tolist()],[counter for ob in item[1]], 'r-x', markersize=2)
                # partial fillings or full fillings
                elif (len(item[1]) > 2) | ((lob.ix[item[-1]].Type.iloc[0] != 2) & (lob.ix[item[-1]].Type.iloc[0] != 3)):
                    ax.plot([tt for tt in lob.ix[item[1]].Time.tolist()],[counter for ob in item[1]], 'b-o', markersize=10)
                else:
                    ax.plot([tt for tt in lob.ix[item[1]].Time.tolist()],[counter for ob in item[1]], 'g-o', markersize=5)
        plt.show()
        return None

    def plot_lob_snapshot(self,nlevels, hour, minute=None, second=None, minVol=0, figsize=(20,15), orderbook=None):
        """
        Displays the evolution of bid and ask quotes at nlevels at the given time intervals
        :param nlevels: Number of levels to display
        :param hour: list -> hour range. int -> hour
        :param minute: list -> minute range. int -> minute
        :param second: list -> second range. int -> second
        :param minVol: ignores all orders for which volume < minVol
        :param figsize: size of figure
        :param orderbook: the orderbook from which we get the quotes (usually self.orderbook after self.load_book(..)
        """
        lob_ask = None
        lob_bid = None
        if orderbook is None:
            orderbook = self.orderbook

        # Set colors
        raws = [(1-i/nlevels) for i in range(0,nlevels) if i<nlevels]
        reds = [(r,0,0) for r in raws]
        blues = [(0,0,b) for b in raws]
        ### bunch of column indices
        askCols = [acol for acol in orderbook.columns if (acol.startswith("ASK") | acol.startswith("Time"))]
        bidCols = [bcol for bcol in orderbook.columns if (bcol.startswith("BID") | bcol.startswith("Time"))]
        bidp = []
        bids = []
        askp = []
        asks = []
        for col in orderbook.columns:
            if col.startswith("BIDp"):
                m = re.findall(r'\d+$', col)
                if int(m[0])<=nlevels:
                    bidp.append(col)

            elif col.startswith("ASKp"):
                m = re.findall(r'\d+$', col)
                if int(m[0])<=nlevels:
                    askp.append(col)

            elif col.startswith("BIDs"):
                m = re.findall(r'\d+$', col)
                if int(m[0])<=nlevels:
                    bids.append(col)

            elif col.startswith("ASKs"):
                m = re.findall(r'\d+$', col)
                if int(m[0])<=nlevels:
                    asks.append(col)

        askCols.append("Timestamp")
        bidCols.append("Timestamp")
        if minute is None:
            if isinstance(hour, int):
                lob_ask = orderbook.loc[orderbook.Hour==hour, askCols]
                lob_bid = orderbook.loc[orderbook.Hour==hour, bidCols]
            elif isinstance(hour, list):
                assert len(hour)==2
                lob_ask = orderbook.loc[(orderbook.Hour>=hour[0]) &
                                        (orderbook.Hour<=hour[1]), askCols]
                lob_bid = orderbook.loc[(orderbook.Hour>=hour[0]) &
                                        (orderbook.Hour<=hour[1]), bidCols]

        elif isinstance(minute, int):
            if second is None:
                lob_ask = orderbook.loc[(orderbook.Hour==hour) &
                                        (orderbook.Minute==minute) &
                                        (orderbook.ASKs1>=minVol),askCols]
                lob_bid = orderbook.loc[(orderbook.Hour==hour) &
                                        (orderbook.Minute==minute) &
                                        (orderbook.BIDs1>=minVol),bidCols]
            elif isinstance(second, int):
                lob_ask = orderbook.loc[(orderbook.Hour==hour) &
                                        (orderbook.Minute==minute) &
                                        (orderbook.Second==second) &
                                        (orderbook.ASKs1>=minVol),askCols]
                lob_bid = orderbook.loc[(orderbook.Hour==hour) &
                                        (orderbook.Minute==minute) &
                                        (orderbook.Second==second) &
                                        (orderbook.BIDs1>=minVol),bidCols]
            elif isinstance(second, list):
                assert len(second)==2
                lob_ask = orderbook.loc[(orderbook.Hour==hour) &
                                        (orderbook.Minute==minute) &
                                        (second[0]<=orderbook.Second) &
                                        (orderbook.Second<=second[1]) &
                                        (orderbook.ASKs1>=minVol),askCols]
                lob_bid = orderbook.loc[(orderbook.Hour==hour) &
                                        (orderbook.Minute==minute) &
                                        (second[0]<=orderbook.Second) &
                                        (orderbook.Second<=second[1]) &
                                        (orderbook.BIDs1>=minVol),bidCols]

        elif isinstance(minute, list):
            assert len(minute)==2
            if second is None:
                lob_ask = orderbook.loc[(orderbook.Hour==hour) &
                                        (minute[0]<=orderbook.Minute) &
                                        (orderbook.Minute<=minute[1]) &
                                        (orderbook.ASKs1>=minVol),askCols]
                lob_bid = orderbook.loc[(orderbook.Hour==hour) &
                                        (minute[0]<=orderbook.Minute) &
                                        (orderbook.Minute<=minute[1]) &
                                        (orderbook.BIDs1>=minVol),bidCols]
            elif isinstance(second, int):
                lob_ask = orderbook.loc[(orderbook.Hour==hour) &
                                        (minute[0]<=orderbook.Minute) &
                                        (orderbook.Minute<=minute[1]) &
                                        (orderbook.Second==second) &
                                        (orderbook.ASKs1>=minVol),askCols]
                lob_bid = orderbook.loc[(orderbook.Hour==hour) &
                                        (minute[0]<=orderbook.Minute) &
                                        (orderbook.Minute<=minute[1]) &
                                        (orderbook.Second==second) &
                                        (orderbook.BIDs1>=minVol),bidCols]
            elif isinstance(second, list):
                assert len(second)==2
                lob_ask = orderbook.loc[(orderbook.Hour==hour) &
                                        (minute[0]<=orderbook.Minute) &
                                        (orderbook.Minute<=minute[1]) &
                                        (second[0]<=orderbook.Second) &
                                        (orderbook.Second<=second[1]) &
                                        (orderbook.ASKs1>=minVol),askCols]
                lob_bid = orderbook.loc[(orderbook.Hour==hour) &
                                        (minute[0]<=orderbook.Minute) &
                                        (orderbook.Minute<=minute[1]) &
                                        (second[0]<=orderbook.Second) &
                                        (orderbook.Second<=second[1]) &
                                        (orderbook.ASKs1>=minVol),bidCols]

        assert lob_bid is not None
        assert lob_ask is not None

        ax = lob_ask.plot(lob_ask.index, askp[0],marker='o',c=reds[0],figsize=figsize, use_index=True)
        for cl in range(1,nlevels):
            lob_ask.plot(lob_ask.index, askp[cl],marker='o',c=reds[cl],figsize=figsize, use_index=True, ax=ax)

        lob_bid.plot(lob_bid.index,bidp[0], marker='s', c=blues[0],ax=ax, use_index=True)
        for cl in range(1,nlevels):
            lob_bid.plot(lob_bid.index,bidp[cl], marker='s', c=blues[cl],ax=ax, use_index=True)
        #lob_bid.plot(lob_bid.index,bidp, marker='s', ax=ax, use_index=True)
        #ax = lob_ask.plot(kind='scatter', x=['Time']*len(asks), y=askp, c=reds, s=50,marker='o', figsize=figsize)
        #lob_bid.plot(kind='scatter', x=['Time']*len(bids), y=bidp, c=blues, s=50,marker='s', ax=ax)

        plt.show()
        return ax

    def aggregate(self, lob=None, frequency='1S', agg_dict=None):
        '''
        Manipulate self.orderbook such that volumes are added and prices are averaged over the period over which we
        aggregate
        :param: lob (pd.DataFrame), the dataframe to be aggregated, if lob is None : lob = self.orderbook (if it already exists)
        :param: frequency (str), e.g. '1S' -> aggregate to 1 second intervals, '5m' -> aggregate to 5 minute intervals
        :param: agg_dict (dict), specifies the rules with which the features are aggregated. Allowed are entries <feature_name> : <aggregation_method>
                aggregation_method could be "sum", "mean", "first", "last",...
        :return: agg_lob which is the aggregated limit order book
        '''
        agg_lob = None

        if (lob is None) & (self.orderbook is not None):
            agg_lob = cp.deepcopy(self.orderbook)
        elif (lob is None) & (self.orderbook is None):
            print("There is no orderbook yet to use. Run method 'load_book' first.")
            return agg_lob
        else:
            agg_lob = lob

        if agg_dict is None:
            agg_dict = {"Time":"first",
                        "Day":"first",
                        "Month":"first",
                        "Year":"first",
                        "Hour":"first",
                        "Minute":"first",
                        "Second":"first",
                        "Millisecond":"first",
                        "Microsecond":"first",
                        "Nanosecond":"first"}
            for col in agg_lob.columns:
                if col.startswith("ASKp") | col.startswith("BIDp"):
                    agg_dict[col] = "mean"
                elif col.startswith("ASKs") | col.startswith("BIDs"):
                    agg_dict[col] = "sum"

        agg_lob = agg_lob.groupby(pd.TimeGrouper(freq=frequency)).agg(agg_dict)
        return agg_lob

    def feature_frame(self, rel_feature_change=False):
        """
        Construct the feature dataframe
        :return:
        """
        feature_columns = ["LastTradeVolume", "LastTradePrice", "Spread", "VolumeImbalance"]

        if not "LastTradeVolume" in self.orderbook.columns:
            self.get_lasttrade_volume()

        if not "LastTradePrice" in self.orderbook.columns:
            self.get_lasttrade_price()

        if not "Spread" in self.orderbook.columns:
            self.orderbook["Spread"] = self.orderbook.ASKp1 - self.orderbook.BIDp1

        if not "VolumeImbalance" in self.orderbook.columns:
            bids_cols = [col for col in self.orderbook.columns if col.startswith("BIDs")]
            asks_cols = [col for col in self.orderbook.columns if col.startswith("ASKs")]
            self.orderbook["VolumeImbalance"] = self.orderbook.loc[:,asks_cols].sum(axis=1) - self.orderbook.loc[:,bids_cols].sum(axis=1)

        if not rel_feature_change:
            return self.orderbook.loc[:, feature_columns]
        else:
            df_feat = self.orderbook.loc[:, feature_columns]
            for col in df_feat.columns:
                df_feat.loc[:,col] = (df_feat.loc[:,col] - df_feat.loc[:,col].shift(1))/df_feat.loc[:,col].shift(1)
            return df_feat

    def plot_volumebalance(self, lt=10, df=None, logscale=True, filename=None):
        """
        Plot ask volumes on lt levels on positive y-axis and bid volumes on lt levels on negative y-axis
        :param lt: level threshold
        :param df: the dataframe from which get volumes. By default this is full orderbook but could also be
        truncated orderbook or time resampled orderbook
        """
        ask_size_cols = [col for col in self.orderbook.columns if (col.startswith("ASKs"))]
        ask_size_cols = [col for col in ask_size_cols if int(re.findall(r'\d+$',col)[0]) <= lt]
        bid_size_cols = [col for col in self.orderbook.columns if col.startswith("BIDs")]
        bid_size_cols = [col for col in bid_size_cols if int(re.findall(r'\d+$',col)[0]) <= lt]

        if df is None:
            df = self.orderbook

        lob_ask = df.loc[:,ask_size_cols]
        lob_bid = df.loc[:,bid_size_cols]

        # build cumulative sum of ask sizes over levels
        #lob_bid = lob_bid.cumsum(axis=1)
        #lob_ask = lob_ask.cumsum(axis=1)
        if logscale:
            lob_bid = np.log(lob_bid)
            lob_ask = np.log(lob_ask)

        lob_bid = -lob_bid
        lob_bidask = pd.concat([lob_bid, lob_ask], axis=1)

        #maxBidLev = "BIDs" + lt.__str__()
        #maxAskLev = "ASKs" + lt.__str__()
        #ymin = lob_bid.loc[:,maxBidLev].min()
        #ymax = lob_ask.loc[:,maxAskLev].max()


        reds = [(1-r/lt,0,0) for r in range(lt)]
        blues = [(0,0,1-b/lt) for b in range(lt)]
        cols = blues + reds

        ax = lob_bidask.plot.area(color=cols, figsize=(20,15))
        title = self.ticker + " " + self.day
        ax.set_ylabel("Bid/Ask Volumes")
        ax.set_xlabel("Time")
        ax.set_title(title)
        if filename is not None:
            fig = ax.get_figure()
            fig.savefig(filename)

        plt.show()
        return ax

    def get_marketorders(self):
        """
        Gets from the orderbook all market orders
        """
        print("Get marketorders...")
        try:
            self.marketorders = self.orderbook.loc[((self.orderbook.Type==4) | (self.orderbook.Type==5))]
        except AttributeError as ae:
            print("There is no orderbook yet. First call 'load_book' method.")
            raise ae

        g = {"Price":"first",
            "Size":"sum",
            "Time":"first",
            "Timestamp":"first",
            "Day":"first",
            "Hour":"first",
            "Minute":"first",
            "Second":"first",
            "Millisecond":"first",
            "Microsecond":"first",
            "Nanosecond":"first",
            "TradeDirection":"first"}
        self.marketorders = self.marketorders.groupby(["Time"]).agg(g)
        self.marketorders = self.marketorders.set_index("Timestamp", drop=False)
        self.marketorders["IntertradeTimes"] = (self.marketorders.Timestamp - self.marketorders.Timestamp.shift(1))
        self.marketorders.loc[:,"IntertradeTimes"] = self.marketorders.IntertradeTimes.astype('timedelta64[ns]')
        self.marketorders.loc[:,"IntertradeTimes"] = self.marketorders.IntertradeTimes.apply(lambda x: self.get_intertrade_times(x,'seconds'))

    def get_lasttrade_durations(self):
        """
        Inserts a column into the orderbook which gives at each time step the time until the last market order
        :return:

        OLD VERSION (MUCH SLOWER):
        print("Getting lasttrade durations.")
        assert self.marketorders is not None
        assert self.orderbook is not None

        self.orderbook["LastTradeDuration"] = 0
        l_tradetime = 0
        for cix in self.marketorders.index:
            c_tradetime = self.marketorders.ix[cix, "Time"]
            self.orderbook.loc[(self.orderbook.Time > l_tradetime) & (self.orderbook.Time <= c_tradetime), "LastTradeDuration"] = \
                c_tradetime - self.orderbook.loc[(self.orderbook.Time > l_tradetime) & (self.orderbook.Time <= c_tradetime), "Time"]
            l_tradetime = c_tradetime
        """
        print("     Get LastTradeDurations...")
        orderbook = self.orderbook
        orderbook["MorderFlag"] = 0
        orderbook.loc[orderbook.Time.isin(self.marketorders.Time), "MorderFlag"] = 1
        orderbook["LastTradeTime"] = orderbook.Time
        orderbook.loc[orderbook.MorderFlag == 0, "LastTradeTime"] = np.nan
        orderbook.iloc[0, orderbook.columns.get_loc("LastTradeTime")] = orderbook.ix[orderbook.LastTradeTime.first_valid_index(), "LastTradeTime"]
        orderbook.loc[:,"LastTradeTime"] = orderbook.loc[:,"LastTradeTime"].fillna(method='pad')
        self.orderbook["LastTradeDuration"] = orderbook.Time - orderbook.LastTradeTime

    def get_lasttrade_volume(self):
        """
        Inserts a column into the orderbook which gives at each time step the last traded volume
        :return:
        """
        print("Getting lasttrade volumes.")
        assert self.marketorders is not None
        assert self.orderbook is not None

        self.orderbook["LastTradeVolume"] = 0
        l_tradetime = 0
        for cix in self.marketorders.index:
            c_tradetime = self.marketorders.ix[cix,"Time"]
            self.orderbook.loc[(self.orderbook.Time > l_tradetime) & (self.orderbook.Time <= c_tradetime), "LastTradeVolume"] = \
                self.marketorders.ix[cix,"Size"]
            l_tradetime = c_tradetime

    def get_lasttrade_price(self):
        """
        Inserts a column into the orderbook which gives at each time step the last traded price
        :return:
        """
        print("Getting lasttrade price.")
        assert self.marketorders is not None
        assert self.orderbook is not None

        self.orderbook["LastTradePrice"] = 0
        l_tradetime = 0
        for cix in self.marketorders.index:
            c_tradetime = self.marketorders.ix[cix,"Time"]
            self.orderbook.loc[(self.orderbook.Time > l_tradetime) & (self.orderbook.Time <= c_tradetime), "LastTradePrice"] = \
                self.marketorders.ix[cix,"Price"]
            l_tradetime = c_tradetime

    def plot_inter_marketorders_time_ecdf(self,show=True):
        """
        Plot the the empirical distribution function of the time gaps between marketorders in self.marketorders
        :return Return axis
        """
        ecdf = ECDF(self.marketorders.IntertradeTimes.values)

        if show:
            fig = plt.figure(figsize=(10,7))
            ax = fig.add_subplot(111)
            ax.ticklabel_format(axis='y', style='sci', scilimits=(-2,-1))
            ax.semilogx(ecdf.x, ecdf.y)
            plt.show()
        return ecdf

    def plot_inter_limitorder_time_ecdf(self,show=True):
        """
        Plot the the empirical cumulative distribution function of the time gaps between marketorders in self.marketorders
        :return Return axis
        """
        self.limitorders = self.orderbook.loc[self.orderbook.Type==1]
        self.limitorders["IntertradeTimes"] = (self.limitorders.loc[:,"Timestamp"] - self.limitorders.loc[:,"Timestamp"].shift(1))
        self.limitorders.loc[:,"IntertradeTimes"] = self.limitorders.IntertradeTimes.astype('timedelta64[ns]')
        self.limitorders.loc[:,"IntertradeTimes"] = self.limitorders.IntertradeTimes.apply(lambda x: self.get_intertrade_times(x,'seconds'))

        ecdf = ECDF(self.limitorders.IntertradeTimes.values)

        if show:
            fig = plt.figure(figsize=(10,7))
            ax = fig.add_subplot(111)
            ax.ticklabel_format(axis='y', style='sci', scilimits=(-2,-1))
            ax.semilogx(ecdf.x, ecdf.y)
            plt.show()
        return ecdf

    def get_limitorder_submissions(self):
        print("     Get limitorder submissions...")
        def min_or_empty(x):
            y = x.nonzero()
            if len(y[0])==0:
                return -1
            else:
                return np.min(y)

        def modulo10_or_nan(x):
            if np.isnan(x):
                return x
            else:
                return x % 10


        self.limitorders = None
        if self.orderbook is None:
            print("first load orderbook ('load orderbook')")
        else:

            bidp_cols = [col for col in self.orderbook.columns if col.startswith("BIDp")]
            askp_cols = [col for col in self.orderbook.columns if col.startswith("ASKp")]
            bids_cols = [col for col in self.orderbook.columns if col.startswith("BIDs")]
            asks_cols = [col for col in self.orderbook.columns if col.startswith("ASKs")]
            bid_cols = bidp_cols + bids_cols
            ask_cols = askp_cols + asks_cols

            df1 = self.orderbook.loc[:,bidp_cols + askp_cols] - self.orderbook.loc[:,bidp_cols + askp_cols].shift(1)
            df1 = df1.apply(lambda x: min_or_empty(x), axis=1)

            df2 = self.orderbook.loc[:,bids_cols + asks_cols] - self.orderbook.loc[:,bids_cols + asks_cols].shift(1)
            df2 = df2.apply(lambda x: min_or_empty(x), axis=1)

            self.orderbook["lo_levelp"] = df1.loc[:]
            self.orderbook["lo_levels"] = df2.loc[:]
            self.orderbook.loc[self.orderbook.lo_levelp != -1, "lo_levels"] = -1
            self.orderbook.loc[self.orderbook.lo_levelp > 9, "lo_levelp"] = self.orderbook.loc[self.orderbook.lo_levelp > 9, "lo_levelp"].apply(lambda x: modulo10_or_nan(x))
            self.orderbook.loc[self.orderbook.lo_levels > 9, "lo_levels"] = self.orderbook.loc[self.orderbook.lo_levels > 9, "lo_levels"].apply(lambda x: modulo10_or_nan(x))
            self.orderbook.loc[self.orderbook.lo_levelp !=-1,"lo_levelp"] += 1
            self.orderbook.loc[self.orderbook.lo_levels !=-1,"lo_levels"] += 1
            self.orderbook["lo_level"] = self.orderbook[["lo_levelp", "lo_levels"]].max(axis=1)

            self.limitorders = self.orderbook.loc[self.orderbook.Type==1]
            # We only record those limit orders that changed something on the levels <= self.nlevels_keep
            self.limitorders = self.limitorders.loc[~((self.limitorders.lo_levelp==-1)&(self.limitorders.lo_levels==-1))]
            self.limitorders["IntertradeTimes"] = (self.limitorders.loc[:,"Timestamp"] - self.limitorders.loc[:,"Timestamp"].shift(1))
            self.limitorders.loc[:,"IntertradeTimes"] = self.limitorders.IntertradeTimes.astype('timedelta64[ns]')
            self.limitorders.loc[:,"IntertradeTimes"] = self.limitorders.IntertradeTimes.apply(lambda x: self.get_intertrade_times(x,'seconds'))

    def get_next_TED(self,prev_TED_start, prev_TED, c, LiquMeasure):

        flag = False
        try:
            next_TED_start = self.orderbook.loc[(self.orderbook.Time > prev_TED_start + prev_TED) & (self.orderbook.loc[:,LiquMeasure] >= c),"Time"].iloc[0]
            next_TED = self.orderbook.loc[(self.orderbook.Time > next_TED_start) & (self.orderbook.loc[:,LiquMeasure] < c),"Time"].iloc[0] - next_TED_start
            return [next_TED_start, next_TED, flag]

        except IndexError as ie:
            print("Reached end of time series")
            flag = True
            return [-1,-1, flag]

    def get_TEDs(self, c=None, LiquMeasure="Spread"):
        """
        Compute the threshold exeedance duration for a given liquidity measure.
        You can choose between Spread and XLM.
        :param prev_TED_start: the previous starting time of an illiquidity threshold exceedance
        :param prev_TED: the previous duration of an illiquidity threshold exeedance
        :param c: threshold
        :param LiquMeasure: liquidity measure (spread or xlm)
        :return:
        """
        print("Computing threshold exceedance durations for measure ", LiquMeasure)
        assert LiquMeasure in ["Spread", "XLM"]

        TED_starts = [0]
        TED = [0]
        flag = False

        if self.orderbook is None:
            print("First load book.")
            return TED_starts, TED

        if "Spread" not in self.orderbook.columns:
            self.orderbook["Spread"] = self.orderbook.ASKp1 - self.orderbook.BIDp1

        if (LiquMeasure == "XLM") and "XLM" not in self.orderbook.columns:
            # compute the xlm measure where we assume that we want to buy and sell (roundtrip) the whole volume
            # in the entire book (per time step)
            bidask_cols = [col for col in self.orderbook.columns if (col.startswith("BID")) | (col.startswith("ASK"))]
            ask_cols = [col for col in self.orderbook.columns if (col.startswith("ASK"))]
            asks_cols = [col for col in self.orderbook.columns if (col.startswith("ASKs"))]
            bid_cols = [col for col in self.orderbook.columns if (col.startswith("BID"))]
            bids_cols = [col for col in self.orderbook.columns if (col.startswith("BIDs"))]

            df = self.orderbook.loc[:, bidask_cols]
            df["p_mid"] = self.orderbook.BIDp1 + (self.orderbook.ASKp1 - self.orderbook.BIDp1)/2
            for clev in range(1,self.nlevels_keep+1):
                curraskp = "ASKp" + clev.__str__()
                currasks = "ASKs" + clev.__str__()
                currbidp = "BIDp" + clev.__str__()
                currbids = "BIDs" + clev.__str__()
                df.loc[:,curraskp] = (df.loc[:,curraskp] - df.loc[:,"p_mid"]) * df.loc[:,currasks]
                df.loc[:,currbidp] = (df.loc[:,currbidp] - df.loc[:,"p_mid"]) * df.loc[:,currbids]
            self.orderbook["XLM"] = df.loc[:,ask_cols].sum(axis=1)/df.loc[:,asks_cols].sum(axis=1) + \
                                    df.loc[:,bid_cols].sum(axis=1)/df.loc[:,bids_cols].sum(axis=1)


        if c is None:
            c = np.percentile(self.orderbook.loc[:,LiquMeasure].values, q=90)


        while not flag:
            next_TED_start, next_TED, flag = self.get_next_TED(TED_starts[-1], TED[-1], c, LiquMeasure)
            TED_starts.append(next_TED_start)
            TED.append(next_TED)

        return TED_starts, TED

    def get_order_age(self, orderbook=None):
        """
        Computes the age for each orderID
        """
        self.orderbook["order_age"] = 0
        unique_loid = self.orderbook.loc[self.orderbook.Type==1, "OrderID"].unique()
        print(len(unique_loid))
        counter = 0
        for id in unique_loid:
            counter += 1
            if counter % 1000 == 0:
                print(counter)
            tmp = self.orderbook.loc[self.orderbook.OrderID == id, "Time"].values
            # if the starting time of the current order is
            if len(tmp) == 1:
                self.orderbook.loc[self.orderbook.OrderID == id, "order_age"] = tmp[-1] - 35100 # 35100 seconds corresponds to 09:45:00
            else:
                self.orderbook.loc[self.orderbook.OrderID == id, "order_age"] = tmp[-1] - tmp[0]


if __name__ == "__main__":

    # SPECS
    ticker = 'AAPL'
    starttime = '34200000'
    endtime = '57600000'
    nlevels = '100'
    local_lob = "/Users/mh/Documents/CSML/Masterarbeit/Python/pickledLobster/"
    rem_lob = "/Volumes/INTENSO/LOBSTER/pickledLobster/"



    print("Loading Lobster for 2014-06-05")
    day = "2014-06-05"
    fname = ticker + "_" + day + "_" + starttime + "_" + endtime + "_" + "lobsterclass_" + nlevels + ".pickle"
    Lobster06 = pickle.load(open(rem_lob + fname, "rb"))
    Lobster06.orderbook = Lobster06.orderbook.reset_index(drop=True)
    print("Getting orderlives...")
    s = Lobster06.get_orderlives(orderbook=Lobster06.orderbook, mode='aggr', legit_type=[1,4,5])
    print(len(s))
