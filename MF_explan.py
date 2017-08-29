import numpy as np
import os
import sys
import lmafit as lmafit
import pandas as pd

class MF_explan():

    def __init__(self, n_movies=50, data_dir='Data'):

        # get ratings
        df = pd.read_csv('{}/ratings.csv'.format(data_dir))

        # create a dataframe with movie IDs on the rows
        # and user IDs on the columns
        ratings = df.pivot(index='movieId', columns='userId', values='rating')

        # put movie titles as index on columns
        movies = pd.read_csv('{}/movies.csv'.format(data_dir))
        movieSeries = pd.Series(list(movies['title']),
                                 index=movies['movieId'])
        ratings = ratings.rename(index=movieSeries)

        # select the 50 movies that have the most ratings
        num_ratings = (~ratings.isnull()).sum(axis=1)
        rows = num_ratings.nlargest(n_movies)
        ratings = ratings.loc[rows.index]

        # eliminate the users that have no ratings in this set
        null_columns = ratings.isnull().all(axis=0)
        null_column_ids = null_columns.index[null_columns]
        ratings = ratings.drop(null_column_ids, axis=1)

        self.ratings = ratings

    def fit_model(self, estimation_rank = 4):
        '''fit the matrix factorization model to the data
        and return a dataframe of predictions.  fitting by
        lmafit uses estimation_rank (default is 4)'''

        m, n = self.ratings.shape

        # put data into form needed by lmafit
        X = self.ratings.values
        known_elements   = np.where(~np.isnan(X))
        list_of_known_elements = zip(*known_elements)
        data = [X[coordinate] for coordinate in list_of_known_elements]

        # run matrix factorization
        self.U, self.V, opts = lmafit.lmafit_mc_adp(m,
                                                  n,
                                                  estimation_rank,
                                                  known_elements,
                                                  data,
                                                  0)
        self.W = ~np.isnan(self.ratings)
        self.pred = pd.DataFrame(self.U.dot(self.V),
                                 index = self.ratings.index,
                                 columns = self.ratings.columns)

        return self.pred

    def jacobian(self, user):
        '''compute the jacobian of the matrix factorization
        for given user, returned as a dataframe'''

        Wj = np.diag(np.where(self.W[user],1,0))
        Jj = pd.DataFrame(self.U @ np.linalg.inv(self.U.T @ Wj @ self.U) @ self.U.T @ Wj,
            index = self.ratings.index,
            columns = self.ratings.index)
        return Jj

    def rated_items(self, user):
        return self.W.index[self.W[user]]

    def num_rated_items_per_user(self):
        return np.sum(self.W, axis=0)
    
    def predictions(self, user):
        '''return the predictions for this user in sorted order'''

        # items that the user has not rated
        unrated = self.ratings.isnull()[user]

        # predictions for those items
        user_pred = self.pred[user][unrated]

        # the highest rated item
        return user_pred.sort_values(ascending=False)

    def report_recommendation(self, J_j, user, recom, silent=False):
        '''present a report that explains the recommendation'''
        
        user_ratings = self.ratings[user]
        rated_status = ~user_ratings.isnull()
        rated_items = user_ratings[rated_status].index

        # construct list of movies, their influence, known rating,
        # and their impact
        sum_of_impacts = np.sum(J_j.loc[recom, rated_items] *
                                self.ratings.loc[rated_items, user])
        influence_list = [(item,
                           J_j.loc[recom, item],
                           self.ratings.loc[item, user],
                           J_j.loc[recom, item] * self.ratings.loc[item, user])
                           for item in rated_items]

        report = pd.DataFrame(np.array(
            [[J_j.loc[recom, item],
                    self.ratings.loc[item, user],
                    J_j.loc[recom, item] * self.ratings.loc[item, user]]                           for item in rated_items]),
                    index = rated_items,
                    columns = ['Influence', 'Known Rating', 'Impact'])

        if silent:
            return report

        print("User ID: {:d}".format(user))
        print("Recommended movie: {:s}".format(recom))
        print("Predicted rating: {:.2f}".format(self.pred.loc[recom, user]))
        print("Number of rated movies: {:d}".format(len(rated_items)))
        rpt_heading = "{:35s} {:12s} {:14s} {:12s}"
        rpt_row = "{:35s} {:9.3f} {:13.1f} {:9.3f}"
        print("\n----------------------------------------------------------------------------")
        print(rpt_heading.format("Rated Movie","Influence","Known rating","Impact"))
        print("----------------------------------------------------------------------------")
        for item in sorted(influence_list, key= lambda L:L[1], reverse = True):
            print(rpt_row.format(item[0][:34],item[1],item[2],item[3]))
        print("----------------------------------------------------------------------------")
        print("{:70s} {:.3f}".format("sum:",sum_of_impacts))

        return report
