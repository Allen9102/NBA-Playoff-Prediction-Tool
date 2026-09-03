# NBA-Playoff-Prediction-Tool
This is a team project to predict NBA playoff performance by converting player-interaction data into heatmap images and feeding them into a dual-branch CNN (ResNet-18).

The project has not been finalized yet; please wait a little longer.

## I. Convert time format: parse_min
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
## II. Create heat maps to analyze the effectiveness of player cooperation
This process analyzes the effectiveness of player cooperation within a single team. It calculates the effectiveness of pairwise combinations based on the players' "simultaneous playing time" and "plus-minus performance" on the court, and finally uses a heatmap to visualize the results for CNN machine learning.

### 1. Provide weights for each team's match results
We set the weight based on the stage that each team reaches. Once a team reaches the round of 16, each of them will have a weight of one point, and each time you advance to the next round, the weight will be multiplied by two until the championship, where it will be 16 points.

### 2. Create heatmap
We have a big for loop to create heatmaps. This process takes time to complete because the NBA API blocks users who request data too frequently; this can be avoided by using `time.sleep(0.6)`.
Moreover, if you fail to produce a heatmap midway, the code will continue producing images. If you rerun the code, the heatmaps previously generated will not be produced again.

### 3. Create a CSV file to save teams' information
We transform the playoff results into a flat, structured DataFrame, with each row representing a team's performance in a given season. We then sort the data by season and playoff wins, and save the final table as a CSV file `labels.csv` for further analysis and modeling.

## III. Machine Learning

### 1. Prepare materials for machine learning
We define a PyTorch dataset for a dual-image input model using NBA heatmap visualizations.
Steps:
1. Read label data from a CSV file
2. Load corresponding heatmap images (shared minutes and plus-minus per minute) for each team-season entry
3. Apply image transformations

### 2. Split training and testing sets based on specific seasons
Before doing machine learning, we split the 2016 ~ 2024 data into training and test groups at a 2:1 ratio, and use the 2024 ~ 2025 regular season data as the prediction object

The result is a clean dataset ready for training deep learning models to predict playoff success based on visual team dynamics.

### 3. Build the model
We use a dual-branch CNN architecture to process two different but complementary visual features:
* The first branch processes the heat map of players' **shared playing time** (measures the closeness of the appearance combination)
* The second branch processes the **plus-minus** heat map per minute (measures the effectiveness of the combination)

The outputs of the two are concatenated and fed into the fully connected layer, and the prediction is a continuous value (number of playoff wins), which meets the requirements of the regression task.

Steps:
1. The DualCNN model uses two pretrained ResNet-18 branches to extract features from two input images.
2. It replaces each ResNet's final layer with nn.Identity() to get 512-dimensional feature vectors, which are concatenated into a 1024-dimensional vector.
3. The combined vector is passed through custom fully connected layers for prediction (e.g., binary classification).
4. The model uses Kaiming initialization and runs on GPU if available; otherwise, it defaults to CPU.
