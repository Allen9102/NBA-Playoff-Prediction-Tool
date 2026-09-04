# NBA-Playoff-Prediction-Tool
This is a team project to predict NBA playoff performance by converting player-interaction data into heatmap images and feeding them into a dual-branch CNN (ResNet-18).

## Methodology overview
```python
NBA API
   ↓
Game & Player Data
   ↓
Shared Minutes + Plus-Minus
   ↓
Heatmaps
   ↓
Dual-Branch ResNet-18
   ↓
Predicted Playoff Wins
   ↓
Softmax Normalization
   ↓
Estimated Championship Probability
```

## I. Convert time format: parse_min
To prepare for machine learning, we have to convert the player's playing time, a string in "mm:ss" format, to a floating-point number of minutes, e.g., "12:30" → 12.5 minutes.
The logic is as follows:
1. If the number is NaN, return 0.0
2. If the value is originally a number, convert it to a floating-point number
3. Split the string "mm:ss" into two integers and convert to minutes: 12:30 → 12 + 30 / 60 → 12.5
4. If the string cannot be split correctly (e.g., malformed, empty string), return 0.0 conservatively.

## II. Create heat maps to analyze the effectiveness of player cooperation
This process analyzes the effectiveness of player cooperation within a single team. It calculates the effectiveness of pairwise combinations based on the players' "simultaneous playing time" and "plus-minus performance" on the court, and finally uses a heatmap to visualize the results for CNN machine learning.

### 1. Provide weights for each team's match results
We set the weight based on the stage that each team reaches. Teams that reach the First Round receive a weight of 1, and the weight doubles for each subsequent playoff round, resulting in a weight of 16 for the champion.

### 2. Filter Eligible Players
To reduce noise from players with limited playing time, we include only players who accumulated at least 820 minutes during the regular season.

### 3. Create heatmap
We construct two heatmaps to represent player cooperation within each team:

* **Shared Minutes:** the estimated amount of time two players were simultaneously on the court in each game.
* **Shared Plus-Minus per Minute:** a measure of the combined plus-minus performance of two players relative to their shared playing time.

For each game, the shared playing time of a player pair is calculated as the minimum of their individual playing times:

`shared_minutes = min(player_1_minutes, player_2_minutes)`
<img width="1200" height="1000" alt="Atlanta_Hawks_201617_shared_minutes" src="https://github.com/user-attachments/assets/c4e93da6-ce4e-4eed-a633-40942d20dd84" />

The shared plus-minus is calculated by summing the individual plus-minus values of the two players:

`shared_PM = player_1_plus_minus + player_2_plus_minus`

These values are aggregated across all games in which the player pair appeared. The shared plus-minus per minute is then calculated as:

`shared_PM_per_min = total_shared_PM / total_shared_minutes`
<img width="1200" height="1000" alt="Atlanta_Hawks_201718_shared_pm_per_min" src="https://github.com/user-attachments/assets/0e31bac9-e9cb-436a-a5c7-ab0e22eeb1d5" />
<img width="1200" height="1000" alt="Boston_Celtics_202324_shared_pm_per_min" src="https://github.com/user-attachments/assets/d652cd47-a34c-42b6-ba9b-c7251c91d2ec" />

The resulting matrices are visualized as heatmaps and used as the two input modalities for the dual-branch CNN.


### 4. Create a CSV file to save teams' information
We transform the playoff results into a flat, structured DataFrame, with each row representing a team's performance in a given season. We then sort the data by season and playoff wins, and save the final table as a CSV file `labels.csv` for further analysis and modeling.

## III. Machine Learning

### 1. Prepare materials for machine learning
We define a PyTorch dataset for a dual-image input model using NBA heatmap visualizations.
Steps:
1. Read label data from a CSV file
2. Load corresponding heatmap images (shared minutes and plus-minus per minute) for each team-season entry
3. Apply image transformations

### 2. Split training and testing sets based on specific seasons
We use four historical seasons (2016–17, 2017–18, 2018–19, and 2021–22) as the training set and two subsequent seasons (2022–23 and 2023–24) as the validation set. The 2024–25 season is reserved as the prediction target.

We split the data by season rather than randomly across team-season samples, allowing the model to be evaluated on seasons that are separate from those used for training.

### 3. Build the model
We use a dual-branch CNN architecture to process two different but complementary visual features:
* The first branch processes the heat map of players' **shared playing time** (measures the closeness of the appearance combination)
* The second branch processes the **plus-minus** heat map per minute (measures the effectiveness of the combination)

The outputs of the two are concatenated and fed into the fully connected layer, and the prediction is a continuous value (number of playoff wins), which meets the requirements of the regression task.

Steps:
1. Two ImageNet-pretrained ResNet-18 models are used as feature extractors.
2. It replaces each ResNet's final layer with `nn.Identity()` to get 512-dimensional feature vectors, which are concatenated into a 1024-dimensional vector.
3. The combined 1024-dimensional feature vector is passed through fully connected layers to predict the number of playoff wins as a continuous value.
4. The model uses Kaiming initialization and runs on GPU if available; otherwise, it defaults to CPU.

### 4. Training
* The model is trained using MSE loss and the AdamW optimizer over up to 600 epochs.
* Training and evaluation are done using separate dataloaders, and losses are recorded each epoch.
* The best model (with the lowest validation loss) is saved, and early stopping is triggered if no improvement occurs for 100 epochs.
* Training and evaluation steps include sending data to the proper device (CPU or GPU) and computing average losses.
<img width="556" height="443" alt="Loss curve" src="https://github.com/user-attachments/assets/91f59234-c5bc-4805-b55a-6c7b54ff26b4" />


## IV. Visiualize data
The model predicts the expected number of playoff wins for each team. For historical playoff seasons, we apply the trained model to the 16 teams that advanced to the first round. The predicted playoff-win values are then converted into relative championship probabilities using softmax normalization across the teams.
<img width="1040" height="634" alt="截圖 2026-09-04 11 06 35" src="https://github.com/user-attachments/assets/1aebeea7-3a04-4c93-a355-ec2fb5f64548" />

The resulting probabilities are saved as CSV files and visualized using bar charts. An interactive dropdown menu allows users to select a season and compare the estimated championship probabilities of the playoff teams. The team with the highest predicted probability is highlighted as the predicted champion.
<img width="455" height="558" alt="截圖 2026-09-04 11 07 20" src="https://github.com/user-attachments/assets/21b427e6-9928-4ed7-881b-44949d8a46b0" />
<img width="1007" height="580" alt="截圖 2026-09-04 11 07 36" src="https://github.com/user-attachments/assets/9853e72b-bb63-4907-a33d-ab2f6df8df50" />


## V. Room for improvement
**1. Automate playoff team selection.**
The playoff team lists are currently hard-coded for each season, such as playoffs_22_23 and playoffs_23_24. Since playoff teams vary from season to season, these lists should be generated automatically from the data.

**2. Parameterize the season of interest.**
We should use a single variable to specify the season of interest and replace hard-coded season references throughout the code. For example, we could define season_of_interest = "2024-25" and use this variable wherever the target season is referenced. This would allow us to change the prediction year by modifying only one variable.

**3. Improve the training and validation strategy.**
Since the goal is to predict future playoff performance, a strictly chronological training and validation split could better reflect the real-world prediction setting.

**4. Improve probability estimation.**
The current championship probabilities are obtained by applying softmax normalization to the model's predicted playoff-win values. Future versions could explore a more statistically grounded approach for estimating and calibrating championship probabilities.
