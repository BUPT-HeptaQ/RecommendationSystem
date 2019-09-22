# based on ALS recommendation system in Spark, use the movies score data in MovieLens to recommend

import sys
import itertools

from math import sqrt
from operator import add
from os.path import join, isfile, dirname

from pyspark import SparkConf, SparkContext
from pyspark.mllib.recommendation import ALS


def parse_rating(line):
    """
    the score format of MovieLens is userId::movieId::rating::timestamp
    """
    fields = line.strip().split("::")
    return float(fields[3]) % 10, (int(fields[0]), int(fields[1]), float(fields[2]))


def parse_movie(line):
    """
    the corresponding format of movie file is movieId::movieTitle
    analysis as int id, txt
    """
    fields = line.strip().split("::")
    return int(fields[0]), fields[1]


def load_ratings(rating_file):
    """
    loading scores
    """
    if not isfile(rating_file):
        print("File %s does not exist." % rating_file)
        sys.exit(1)
    f = open(rating_file, 'r')
    ratings = filter(lambda r: r[2] > 0, [parse_rating(line)[1] for line in f])
    f.close()

    if not ratings:
        print("No ratings are provided.")
        sys.exit(1)
    else:
        return ratings


def compute_RMSE(model, data, n):
    """
    calculate root-mean-square error used to evaluate the recommendation
    """
    prediction_RMSE = model.predictAll(data.map(lambda x: (x[0], x[1])))
    predictions_and_ratings = prediction_RMSE.map(lambda x: ((x[0], x[1]), x[2])).\
        join(data.map(lambda x: ((x[0], x[1]), x[2]))).values()

    return sqrt(predictions_and_ratings.map(lambda x: (x[0] - x[1]) ** 2).reduce(add) / float(n))


if __name__ == "__main__":
    if(len(sys.argv) != 3):
        print("Usage: /path/to/spark/bin/spark-submit --driver-memory 2g " +
              "MovieLensALS.py movieLensDataDir personalRatingsFile")
        sys.exit(1)

    # set the environment
    conf = SparkConf().setAppName("MovieLensALS").set("spark.executor.memory", "2g")
    sc = SparkContext(conf=conf)

    # loading the score data
    my_ratings = load_ratings(sys.argv[2])
    my_ratings_RDD = sc.parallelize(my_ratings, 1)

    movie_lens_home_dir = sys.argv[1]

    # get the format of ratings is RDD (timestamp is the last digit, (useId, movieId, rating))
    ratings = sc.textFile(join(movie_lens_home_dir, "ratings.dat")).map(parse_rating)

    # get the format of movies is RDD (movieId, movieTitle)
    movies = dict(sc.textFile(join(movie_lens_home_dir, "movies.dat")).map(parse_movie).collect())

    num_ratings = ratings.count()
    num_users = ratings.value().map(lambda r: r[0]).distinct().count()
    num_movies = ratings.values().map(lambda r: r[1]).distinct().count()

    print("Got %d ratings from %d user on %d movies." % (num_ratings, num_users, num_movies))
    # according to the last digit of timestamp, divide the whole data set: train data set(60%),
    # cross validation set (20%) and evaluate set (20%). all the data set format is RDD (userId, movieId, rating)

    num_partition = 4
    training = ratings.filter(lambda x: x[0] < 6).values().union(my_ratings_RDD).repartition(num_partition).cache()

    validation = ratings.filter(lambda x: x[0] >= 6 & x[0] < 8).values().repartition(num_partition).cache()

    test = ratings.filter(lambda x: x[0] >= 8).values().cache()

    num_training = training.count()
    num_validation = validation.count()
    num_test = test.count()

    print("Training: %d, validation:%d, test: %d" % (num_training, num_validation, num_test))

    # training the model and check the outcomes on cross validation data set

    ranks = [8, 12]
    lambdas = [0.1, 10.0]
    num_iters = [10, 20]
    best_model = None
    best_validation_RMSE = float("inf")
    best_rank = 0
    best_lambda = -1.0
    best_num_iter = -1

    for rank, lmbda, num_iter in itertools.product(ranks, lambdas, num_iters):
        model = ALS.train(training, rank, num_iter, lmbda)
        validation_RMSE = compute_RMSE(model, validation, num_validation)
        print("RMSE(validation) = %f for the model trained with" % validation_RMSE +
              "rank = %d, lambda = %.lf, and num_iter = %d" % (rank, lmbda, num_iter))

        if (validation_RMSE < best_validation_RMSE):
            best_model = model
            best_validation_RMSE = validation_RMSE
            best_rank = rank
            best_lambda = lmbda
            best_num_iter = num_iter

    test_RMSE = compute_RMSE(best_model, test, num_test)

    # evaluate on the test set, find the best model on the cross validation set
    print("The best model was trained with rank = %d and lambda = %.lf," % (best_rank, best_lambda) +
          "and num_iter = %d, and its RMSE on the test set is %f." % (best_num_iter, test_RMSE))

    # setting the baseline model return the model get arg score every times
    mean_rating = training.union(validation).map(lambda x: x[2].mean())
    baseline_RMSE = sqrt(test.map(lambda x: (mean_rating - x[2]) ** 2).reduce(add) / num_test)
    improvement = (baseline_RMSE - test_RMSE) / baseline_RMSE * 100
    print("The best model improves the baseline by %.2f", improvement, "%")

    # personal recommendations (aim at users)
    my_rate_movie_id = set([x[1] for x in my_ratings])
    candidates = sc.parallelize([m for m in movies if m not in my_rate_movie_id])
    predictions = best_model.predictAll(candidates.map(lambda x: (0, x))).collect()
    recommendations = sorted(predictions, key=lambda x: x[2], reverse=True)[:50]

    print("Movies recommend for you:")
    for i in range(len(recommendations)):
        print("%2d: %s" % (i + 1, movies[recommendations[i][1]])).encode('ascii', 'ignore')

    # clean up
    sc.stop()

