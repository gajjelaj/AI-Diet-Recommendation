from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle 
from sklearn.preprocessing import StandardScaler


app = Flask(__name__)


food_df = pd.read_csv(r'E:\Jayanth\project\Dataset\Calories.csv')

# Convert 'Calories' column to numeric
food_df["Calories"] = food_df["Calories"].str.extract(r"(\d+)").astype(float)


training_df = pd.read_csv("Dataset.csv")
# Convert 'Calories' column to numeric
food_df["Calories"] = food_df["Calories"].astype(float)


def calculate_bmi(weight_kg, height_m):
    return weight_kg / (height_m**2)


def calculate_bmr(age, weight_kg, height_m, gender):
    if gender == "F":  # Female
        return 655 + (9.6 * weight_kg) + (1.8 * height_m * 100) - (4.7 * age)
    elif gender == "M":  # Male
        return 66 + (13.7 * weight_kg) + (5 * height_m * 100) - (6.8 * age)
    else:
        raise ValueError("Invalid gender value")


def calculate_daily_calories_direct(
    user_inputs, initial_weight, desired_weight, time_interval_months, model
):
    from sklearn.preprocessing import StandardScaler
    # Extract user inputs
    age = user_inputs["age"]
    weight_kg = user_inputs["weight(kg)"]
    height_m = user_inputs["height(m)"]
    gender = (
        "M" if user_inputs["gender"] == 1 else "F"
    )  # Assuming binary gender variable

    # Calculate BMI and BMR using the provided formulas
    BMI = calculate_bmi(weight_kg, height_m)
    BMR = calculate_bmr(age, weight_kg, height_m, gender)

    # Calculate total weight change
    weight_change = desired_weight - initial_weight
    calories_per_kg = 7700

    # Convert months to days (assuming 30 days per month)
    time_interval_days = time_interval_months * 30

    daily_weight_change = weight_change / time_interval_days  # Daily weight change

    # Estimated calories per kg change

    # Calculate caloric difference based on weight change goal
    caloric_difference = daily_weight_change * calories_per_kg

    # Create input array for model prediction
    input_array = np.array(
        [
            [
                age,
                weight_kg,
                height_m,
                BMI,
                BMR,
                user_inputs["activity_level"],
                user_inputs["gender"],
            ]
        ]
    )

    # Use the trained model to predict daily maintenance calories
    scaler = StandardScaler()
    input_array = scaler.fit_transform(input_array)
    model_predicted_calories = model.predict(input_array)

    # #     # Sum the model predicted calories and caloric difference
    daily_cal = model_predicted_calories + caloric_difference

    return model_predicted_calories, daily_cal

model_filename = 'best_model.pkl'
with open(model_filename, 'rb') as file:
    model = pickle.load(file)

def calculate_multiplication_factor(
    intentional_exercise, moderate_vigorous_activity, time_on_feet
):
    if intentional_exercise == "a":
        if moderate_vigorous_activity == "a" and time_on_feet == "a":
            return 1.2  # Sedentary
        elif moderate_vigorous_activity in ["b", "c", "d"] and time_on_feet in [
            "b",
            "c",
            "d",
        ]:
            return 1.375  # Lightly Active
        else:
            return None
    elif intentional_exercise == "b":
        if moderate_vigorous_activity in ["b", "c"] and time_on_feet in [
            "a",
            "b",
            "c",
            "d",
        ]:
            return 1.375  # Lightly Active
        elif moderate_vigorous_activity == "a" and time_on_feet == "a":
            return 1.2  # Sedentary
        elif moderate_vigorous_activity == "a" and time_on_feet in ["b", "c", "d"]:
            return 1.375  # Lightly Active
        elif moderate_vigorous_activity == "d" and time_on_feet in ["a", "b", "c"]:
            return 1.55  # Moderately Active
        elif moderate_vigorous_activity == "d" and time_on_feet == "d":
            return 1.725  # Very Active
        else:
            return None
    elif intentional_exercise == "c":
        if moderate_vigorous_activity == "a" and time_on_feet == "a":
            return 1.2  # Sedentary
        elif moderate_vigorous_activity == "a" and time_on_feet in ["b", "c", "d"]:
            return 1.375  # Lightly Active
        elif moderate_vigorous_activity in ["b", "d"] and time_on_feet in [
            "a",
            "b",
            "c",
            "d",
        ]:
            return 1.55  # Moderately Active
        elif moderate_vigorous_activity == "c" and time_on_feet == "a":
            return 1.375  # Lightly Active
        elif moderate_vigorous_activity == "c" and time_on_feet in ["b", "c", "d"]:
            return 1.55  # Moderately Active
        elif moderate_vigorous_activity == "d" and time_on_feet in ["a", "b"]:
            return 1.55  # Moderately Active
        elif moderate_vigorous_activity == "d" and time_on_feet in ["c", "d"]:
            return 1.725  # Very Active
        else:
            return None
    elif intentional_exercise == "d":
        if moderate_vigorous_activity == "a" and time_on_feet == "a":
            return 1.375  # Lightly Active
        elif moderate_vigorous_activity == "b" and time_on_feet == "a":
            return 1.375  # Lightly Active
        elif moderate_vigorous_activity == "c" and time_on_feet == "a":
            return 1.375  # Lightly Active
        elif moderate_vigorous_activity == "d" and time_on_feet == "a":
            return 1.55  # Moderately Active
        elif moderate_vigorous_activity == "a" and time_on_feet in ["b", "c", "d"]:
            return 1.375  # Lightly Active
        elif moderate_vigorous_activity == "b" and time_on_feet in ["b", "c", "d"]:
            return 1.375  # Lightly Active
        elif moderate_vigorous_activity == "c" and time_on_feet in ["b", "c", "d"]:
            return 1.55  # Moderately Active
        elif moderate_vigorous_activity == "d" and time_on_feet in ["b", "c"]:
            return 1.725  # Very Active
        elif moderate_vigorous_activity == "d" and time_on_feet in ["d"]:
            return 1.9  # Extra Active
        else:
            return None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/form")
def form():
    return render_template("form.html")


food_and_calories_df = pd.read_csv("Food and Calories.csv")


@app.route("/recommend", methods=["POST"])
def recommend():
    recommendations = ""
    age = int(request.form["age"])
    gender = request.form["gender"]
    weight = float(request.form["weight"])
    height_cm = float(request.form["height"])
    # activity = float(request.form['activity'])
    intentional_exercise = request.form["intentional_exercise"]
    moderate_vigorous_activity = request.form["moderate_vigorous_activity"]
    time_on_feet = request.form["time_on_feet"]

    print(gender)
    activity_level = calculate_multiplication_factor(
        intentional_exercise, moderate_vigorous_activity, time_on_feet
    )
    print(activity_level)

    initial_weight = weight
    desired_weight = float(request.form["desired_weight"])
    time_interval_months = int(request.form["time_interval_months"])
    height_m = height_cm / 100
    # Prepare user inputs dictionary

    meal_choice = int(request.form["meal_choice"])
    user_inputs = {
        "age": age,
        "weight(kg)": weight,
        "height(m)": height_m,
        "gender": 1 if gender == "M" else 0,
        "activity_level": activity_level,
    }
    user_inputs

    # # Calculate daily calorie difference for weight change

    # Calculate daily calorie difference for weight change
    daily_calorie_difference_direct = calculate_daily_calories_direct(
        user_inputs, initial_weight, desired_weight, time_interval_months, model
    )

    import pandas as pd

    your_label_value = daily_calorie_difference_direct[0][0]
    print(your_label_value)
    row = training_df[training_df["Label"] == your_label_value]

    calories_value = row["calories_to_maintain_weight"].values[0]

    food_and_calories_df[food_and_calories_df["Calories (cal)"] == 22]

    required_calories = calories_value
    print(required_calories)
    food_combinations = []
    for i in range(len(food_and_calories_df)):
        for j in range(i + 1, len(food_and_calories_df)):
            total_calories = (
                food_and_calories_df.iloc[i]["Calories (cal)"]
                + food_and_calories_df.iloc[j]["Calories (cal)"]
            )
            food_combinations.append(
                (
                    food_and_calories_df.iloc[i]["Food"],
                    food_and_calories_df.iloc[i]["Serving"],
                    food_and_calories_df.iloc[i]["Calories (cal)"],
                    food_and_calories_df.iloc[j]["Food"],
                    food_and_calories_df.iloc[j]["Serving"],
                    food_and_calories_df.iloc[j]["Calories (cal)"],
                    total_calories,
                )
            )

    # Sort food combinations based on the absolute difference between total calories and required calories
    food_combinations.sort(key=lambda x: abs(x[6] - required_calories))

    # Find three food items whose total caloric value is closest to one-third of the required calorie intake for each meal
    meal_calories = required_calories / 3
    # Sort food combinations based on the absolute difference between total calories and one-third of the required calories
    food_combinations.sort(key=lambda x: abs(x[6] - meal_calories))

    breakfast_proportion = 0.25
    lunch_proportion = 0.45
    dinner_proportion = 0.35

    # Calculate caloric intake for each meal based on the user's total required calories
    breakfast_calories = required_calories * breakfast_proportion
    lunch_calories = required_calories * lunch_proportion
    dinner_calories = required_calories * dinner_proportion

    # Find food items whose total caloric value is closest to one-third of the required calorie intake for each meal
    meal1_calories = breakfast_calories
    meal2_calories = lunch_calories
    meal3_calories = dinner_calories

    # Sort food combinations based on the absolute difference between total calories and one-third of the required calories for each meal
    food_combinations.sort(key=lambda x: abs(x[6] - meal1_calories))
    meal1_combinations = food_combinations[:]

    food_combinations.sort(key=lambda x: abs(x[6] - meal2_calories))
    meal2_combinations = food_combinations[:]

    food_combinations.sort(key=lambda x: abs(x[6] - meal3_calories))
    meal3_combinations = food_combinations[:]

    # Initialize lists to store food combinations for each meal
    breakfast_combinations = []
    lunch_combinations = []
    dinner_combinations = []

    # Initialize variables to track the total calorie count for each meal
    breakfast_calories_total = 0
    lunch_calories_total = 0
    dinner_calories_total = 0

    # Generate combinations of food items only from the filtered subset
    for combination in meal1_combinations:
        total_calories = combination[6]
        if breakfast_calories_total + total_calories <= breakfast_calories:
            breakfast_combinations.append(combination)
            breakfast_calories_total += total_calories

    for combination in meal2_combinations:
        total_calories = combination[6]
        if lunch_calories_total + total_calories <= lunch_calories:
            lunch_combinations.append(combination)
            lunch_calories_total += total_calories

    for combination in meal3_combinations:
        total_calories = combination[6]
        if dinner_calories_total + total_calories <= dinner_calories:
            dinner_combinations.append(combination)
            dinner_calories_total += total_calories

    # Calculate total calories for breakfast, lunch, and dinner
    total_breakfast_calories = sum(meal_comb[6] for meal_comb in breakfast_combinations)
    total_lunch_calories = sum(meal_comb[6] for meal_comb in lunch_combinations)
    total_dinner_calories = sum(meal_comb[6] for meal_comb in dinner_combinations)

    # Calculate the overall total
    overall_total_calories = (
        total_breakfast_calories + total_lunch_calories + total_dinner_calories
    )
    if meal_choice == 1:
        recommendations += "\nFood Recommendations for Breakfast:\n"
        for meal_combination in breakfast_combinations:
            recommendations += f"{meal_combination[0]} ({meal_combination[1]}) - {meal_combination[2]} Calories\n"
            recommendations += f"{meal_combination[3]} ({meal_combination[4]}) - {meal_combination[5]} Calories\n"
            recommendations += "Total Calories: {}\n\n".format(meal_combination[6])
    elif meal_choice == 2:
        recommendations += "\nFood Recommendations for Breakfast:\n"
        for meal_combination in breakfast_combinations:
            recommendations += f"{meal_combination[0]} ({meal_combination[1]}) - {meal_combination[2]} Calories\n"
            recommendations += f"{meal_combination[3]} ({meal_combination[4]}) - {meal_combination[5]} Calories\n"
            recommendations += "Total Calories: {}\n\n".format(meal_combination[6])

        recommendations += "\nFood Recommendations for Lunch:\n"
        for meal_combination in lunch_combinations:
            recommendations += f"{meal_combination[0]} ({meal_combination[1]}) - {meal_combination[2]} Calories\n"
            recommendations += f"{meal_combination[3]} ({meal_combination[4]}) - {meal_combination[5]} Calories\n"
            recommendations += "Total Calories: {}\n\n".format(meal_combination[6])
    elif meal_choice == 3:
        recommendations += "\nFood Recommendations for Breakfast:\n"
        for meal_combination in breakfast_combinations:
            recommendations += f"{meal_combination[0]} ({meal_combination[1]}) - {meal_combination[2]} Calories\n"
            recommendations += f"{meal_combination[3]} ({meal_combination[4]}) - {meal_combination[5]} Calories\n"
            recommendations += "Total Calories: {}\n\n".format(meal_combination[6])

        recommendations += "\nFood Recommendations for Lunch:\n"
        for meal_combination in lunch_combinations:
            recommendations += f"{meal_combination[0]} ({meal_combination[1]}) - {meal_combination[2]} Calories\n"
            recommendations += f"{meal_combination[3]} ({meal_combination[4]}) - {meal_combination[5]} Calories\n"
            recommendations += "Total Calories: {}\n\n".format(meal_combination[6])

        recommendations += "\nFood Recommendations for Dinner:\n"
        for meal_combination in dinner_combinations:
            recommendations += f"{meal_combination[0]} ({meal_combination[1]}) - {meal_combination[2]} Calories\n"
            recommendations += f"{meal_combination[3]} ({meal_combination[4]}) - {meal_combination[5]} Calories\n"
            recommendations += "Total Calories: {}\n\n".format(meal_combination[6])

        return recommendations
    
    print(recommendations)
    
    return render_template("results.html", recommendations=recommendations)


# Load the CSV file into a DataFrame
food_calories_df = food_df


# Function to search for the calorie count of a food item
def search_food_calories(food_name):
    # Convert the food name to lowercase for case-insensitive search
    food_name = food_name.lower()

    # Search for the food item in the DataFrame
    result = food_calories_df[food_calories_df["Food"].str.lower() == food_name]

    # If the food item is found, return its calorie count
    if not result.empty:
        return result["Calories"].values[0]
    else:
        return None  # Return None if the food item is not found


@app.route("/calories")
def calories():
    return render_template("calories.html")


@app.route("/search", methods=["POST"])
def search():
    searched_food = request.form["food"]
    calories = search_food_calories(searched_food)
    if calories is not None:
        return render_template(
            "caloryResult.html", food=searched_food.capitalize(), calories=calories
        )
    else:
        return render_template("cNotFound.html", food=searched_food.capitalize())


if __name__ == "__main__":
    app.run(debug=False)
