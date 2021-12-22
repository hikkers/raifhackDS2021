from catboost import CatBoostRegressor

train_clustered = pd.read_csv("train_clustered.csv")
train_clustered = pd.read_csv("test_clustered.csv")


train_data = train_clustered.drop(["per_square_meter_price", "city", "region", "osm_city_nearest_name", "date"], axis=1)
train_labels = train_clustered["per_square_meter_price"]
eval_data = test_clustered.drop(["city", "region", "osm_city_nearest_name", "date"], axis=1)

model = CatBoostRegressor(iterations=2, learning_rate=1, depth=2)
model.fit(train_data, train_labels)
preds = model.predict(eval_data)

submission = pd.DataFrame(columns=["id", "per_square_meter_price"])
submission["id"] = "COL_" + test_clustered["id"].astype(str)
submission["per_square_meter_price"] = preds
submission.to_csv("submission.csv", index=False)
