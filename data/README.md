# Dataset

On the first run, `train_model.py` creates `used_cars.csv` here with 360
deterministic synthetic examples. These values are for demonstrating data
preparation and regression only; they are not real listings or market prices.

The generated columns cover company, model, year, kilometers driven, fuel,
transmission, owner count category, engine size, power, seats, and the synthetic
target price in Indian rupees. The training code keeps an existing CSV in place
and validates that it contains the expected columns.