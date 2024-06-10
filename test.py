from recommend import generate_recommendations

# User responses dictionary
user_responses = {
    'city': 'toronto',
    'budget': '200',
    'season': 'Summer',
    'time_of_the_day': 'Afternoon',
    'weather': 'Snowy',
    'time': 'Half-day',
    'company': 'Family',
    'destination_and_theme': 'Food'
}

# Extract values for 'city' and 'budget'
user_city = user_responses.pop('city')
user_budget = float(user_responses.pop('budget'))

# Call the generate_recommendations function
recommendations = generate_recommendations(user_responses, user_city, user_budget)

# Display the recommendations on screen
print("Top recommendations:")
for rec in recommendations:
    print(f"Name: {rec['name']}")
    print(f"Location: {rec['location']}")
    print(f"Price: {rec['price']}")
    print(f"Link: {rec['link']}")
    print(f"Score: {rec['score']}\n")
