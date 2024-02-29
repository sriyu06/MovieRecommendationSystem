from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split
from collections import defaultdict

# Load the dataset
reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1)
data = Dataset.load_from_file('ratings.csv', reader=reader)

# Split the dataset into train and test sets
trainset, testset = train_test_split(data, test_size=0.2)

# Build the collaborative filtering model
model = KNNBasic(sim_options={'user_based': True})
model.fit(trainset)

# Get top N movie recommendations for a given user
def get_top_n_recommendations(user_id, n=10):
    test_user_ratings = defaultdict(int)
    for (item_id, rating) in testset:
        if user_id == item_id:
            test_user_ratings[item_id] = rating

    top_n = defaultdict(float)
    for item_id in model.trainset._raw2inner_id_items.keys():
        if item_id not in test_user_ratings:
            inner_id = model.trainset.to_inner_iid(item_id)
            neighbors = model.get_neighbors(inner_id, k=n)
            for neighbor in neighbors:
                movie_id = model.trainset.to_raw_iid(neighbor)
                top_n[movie_id] += model.trainset.ur[user_id][movie_id]

    top_n = sorted(top_n.items(), key=lambda kv: kv[1], reverse=True)[:n]
    return top_n

# Example usage
user_id = '1'
recommendations = get_top_n_recommendations(user_id)
print(f"Top 10 movie recommendations for user {user_id}:")
for movie_id, rating in recommendations:
    print(f"Movie ID: {movie_id}, Predicted Rating: {rating}")
