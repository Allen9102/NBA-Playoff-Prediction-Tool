# NBA-Playoff-Prediction-Tool
This is a team project to predict NBA playoff performance by converting player-interaction data into heatmap images and feeding them into a dual-branch CNN (ResNet-18).

The project has not been finalized yet; please wait for me a while longer.

## Sub function: Convert time format
To prepare for machine learning, we have to convert the player's playing time, a string in "mm:ss" format, to a floating-point number of minutes, e.g., "12:30" → 12.5 minutes.
The logic is as follows:
1. If the number is NaN, return 0.0
2. If the value is originally a number, convert it to a floating-point number
3. Split the string "mm:ss" into two integers and convert to minutes: 12:30 → 12 + 30 / 60 → 12.5
4. If the string cannot be split correctly (e.g., malformed, empty string), return 0.0 conservatively.
```python
def parse_min(min_str):
    if pd.isnull(min_str):
        return 0.0 # if NaN, return 0.0
    if isinstance(min_str, (int, float)):
        return float(min_str) # dim it as a float.
    try:
        minutes, seconds = map(int, min_str.split(':'))
        return minutes + seconds / 60 # mm:ss to a float
    except:
        return 0.0
```
