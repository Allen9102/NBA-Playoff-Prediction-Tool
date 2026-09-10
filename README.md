# NBA-Playoff-Prediction-Tool

A team project that predicts NBA playoff performance by transforming player-interaction data into heatmap representations and using a dual-branch CNN based on ResNet-18.

## Project Structure
```text
NBA-Playoff-Prediction-Tool/
├── shared_minutes/
├── shared_pm_per_min/
├── .gitignore
├── LICENSE
├── NBA Playoff Prediction Tool.py
├── README.md
├── best_dual_model.pth
├── labels.csv
└── nba_prediction_chart.png
```
- `data/heatmap/shared_minutes/`: Heatmaps representing the shared playing time of player pairs.
- `data/heatmap/shared_pm_per_min/`: Heatmaps representing the shared plus-minus per minute of player pairs.
- `data/labels.csv`: Team-season information and corresponding playoff performance used as training labels.
- `model/best_model.pth`: Saved model weights from the best-performing model based on validation loss.
- `results/`: Model outputs, loss curves, championship probability results, and visualizations.
- `NBA Playoff Prediction Tool.py`: Main Python script for data processing, heatmap generation, model training, prediction, and visualization.

## My Contribution

My primary contributions to this project were:

- **Data Visualization:** Designed and analyzed player-interaction heatmaps in collaboration with a teammate.
- **Model Interpretation:** Visualized model predictions and playoff outcomes to make the results easier to interpret.
- **Statistical Communication:** Explained the rationale behind using MSE loss and regression-based evaluation to teammates without a statistics background.

## Methodology Overview

    NBA API
       ↓
    Game & Player Data
       ↓
    Pairwise Player Statistics
       ↓
    Two Heatmap Modalities
       ↓
    Dual-Branch ResNet-18
       ↓
    Predicted Playoff Wins
       ↓
    Softmax Transformation
       ↓
    Estimated Championship Probability

## I. Data Preparation

### 1. Convert Playing Time

Player playing time was originally provided as strings in `mm:ss` format. We converted these values into numerical minutes for downstream feature construction.

The conversion logic handles:

1. Missing values by returning `0.0`
2. Numeric values by converting them to floating-point numbers
3. `mm:ss` strings by converting seconds into fractional minutes
4. Malformed or empty strings conservatively by returning `0.0`

For example:

`12:30 → 12 + 30 / 60 → 12.5`

### 2. Playoff Outcome Weighting

We assigned weights based on the playoff stage reached by each team. Teams eliminated in the First Round receive a weight of 1, and the weight doubles for each subsequent playoff round, resulting in a weight of 16 for the champion.

### 3. Filter Eligible Players

To reduce noise from players with limited playing time, we included only players who accumulated at least 820 minutes during the regular season.

## II. Player-Interaction Heatmaps

We represent player interactions within each team using two complementary pairwise features:

- **Shared Minutes:** captures the extent to which two players overlap in playing time.
- **Shared Plus-Minus per Minute:** captures their combined on-court performance relative to their shared playing time.

### 1. Shared Minutes

For each player pair, shared playing time is approximated as:

`shared_minutes = min(player_1_minutes, player_2_minutes)`

These values are aggregated across games in which the player pair appeared.

<img width="1200" height="1000" alt="Atlanta_Hawks_201617_shared_minutes" src="https://github.com/user-attachments/assets/c4e93da6-ce4e-4eed-a633-40942d20dd84" />

### 2. Shared Plus-Minus per Minute

The shared plus-minus is calculated as:

`shared_PM = player_1_plus_minus + player_2_plus_minus`

The values are aggregated across games, and shared plus-minus per minute is calculated as:

`shared_PM_per_min = total_shared_PM / total_shared_minutes`

<img width="1200" height="1000" alt="Atlanta_Hawks_201718_shared_pm_per_min" src="https://github.com/user-attachments/assets/0e31bac9-e9cb-436a-a5c7-ab2f6df8df50" />

<img width="1200" height="1000" alt="Boston_Celtics_202324_shared_pm_per_min" src="https://github.com/user-attachments/assets/d652cd47-a34c-42b6-ba9b-c7251c91d2ec" />

The resulting pairwise matrices are visualized as heatmaps and used as the two input modalities for the dual-branch CNN.

### 3. Construct Training Labels

We transform playoff results into a structured DataFrame, with each row representing a team's performance in a given season.

The resulting data are sorted by season and playoff wins and saved as `labels.csv` for downstream modeling.

## III. Machine Learning: Dual-Branch CNN Regression

### 1. Prepare the Dataset

We define a PyTorch dataset for a dual-image input model.

The dataset pipeline:

1. Reads team-season labels from a CSV file
2. Loads the corresponding shared-minutes and plus-minus heatmaps
3. Applies image transformations
4. Returns the two heatmaps together with the corresponding playoff-win target

### 2. Temporal Train-Validation Split

We split the data by season rather than randomly across team-season observations.

- **Training:** 2016–17, 2017–18, 2018–19, and 2021–22
- **Validation:** 2022–23 and 2023–24
- **Prediction Target:** 2024–25

This separation allows the model to be evaluated on seasons that are distinct from those used for training.

### 3. Dual-Branch ResNet-18

We use a dual-branch CNN architecture to process two complementary heatmap representations:

- The **shared-minutes branch** represents the degree of overlap in player appearances.
- The **plus-minus-per-minute branch** represents the effectiveness of player combinations.

The two branches independently extract image features. Their outputs are then concatenated and passed through fully connected layers to predict the number of playoff wins as a continuous value.

The model architecture consists of:

1. Two ImageNet-pretrained ResNet-18 models used as feature extractors.
2. Replacement of each ResNet-18 final layer with `nn.Identity()` to obtain a 512-dimensional feature vector.
3. Concatenation of the two feature vectors into a 1024-dimensional representation.
4. Fully connected layers that predict playoff wins as a continuous regression target.
5. Kaiming initialization for the trainable layers.
6. GPU execution when available, with CPU used as a fallback.

### 4. Model Training

The model is trained using:

- **Loss function:** Mean Squared Error (MSE)
- **Optimizer:** AdamW
- **Maximum epochs:** 600
- **Early stopping:** 100 epochs without improvement
- **Model selection:** The model with the lowest validation loss is saved.

Training and validation losses are recorded for each epoch.

<img width="556" height="443" alt="Loss curve" src="https://github.com/user-attachments/assets/91f59234-c5bc-4805-b55a-6c7b54ff26b4" />

## IV. Results and Visualization

The model predicts the expected number of playoff wins for each team.

For historical playoff seasons, we apply the trained model to the 16 teams that advanced to the first round. The predicted playoff-win values are then transformed into relative championship probabilities using softmax normalization across the teams.


<img width="1040" height="634" alt="預測結果（綠色）" src="https://github.com/user-attachments/assets/29ae5f89-cda1-44a3-9f64-7a4c0f2bfc06" />


The resulting probabilities are saved as CSV files and visualized using bar charts. An interactive dropdown menu allows users to select a season and compare the estimated championship probabilities of playoff teams.

The team with the highest estimated probability is highlighted as the predicted champion.


<img width="455" height="558" alt="bar可調整" src="https://github.com/user-attachments/assets/5f83def9-0fd3-4cad-a66d-982d108465e3" />


<img width="1007" height="580" alt="預測結果" src="https://github.com/user-attachments/assets/37009f69-8dce-4d5b-89f8-96856b20b862" />


## V. Limitations and Future Improvements

### 1. Automate Playoff Team Selection

The playoff team lists are currently hard-coded for individual seasons, such as `playoffs_22_23` and `playoffs_23_24`. Future versions could generate playoff team lists automatically from the underlying data to make the prediction pipeline more scalable.

### 2. Parameterize the Target Season

Season-specific references are currently hard-coded in parts of the code. A single parameter such as `season_of_interest = "2024-25"` could be used throughout the pipeline, allowing the prediction target to be changed without modifying multiple sections of the code.

### 3. Strengthen Temporal Validation

The current workflow already separates training, validation, and prediction seasons chronologically. However, the dataset contains a relatively small number of seasons.

Future work could use rolling or expanding-window validation across multiple historical seasons to obtain a more robust estimate of out-of-sample performance.

### 4. Improve Probability Estimation

The current championship probabilities are obtained by applying softmax normalization to the predicted playoff-win values. This provides relative probabilities across the playoff teams but does not directly model the probability of winning the championship.

Future versions could explore more statistically grounded approaches for estimating and calibrating championship probabilities.

### 5. Improve Model Interpretability

The dual-branch CNN provides limited insight into which player interactions contribute most to the predicted playoff performance.

Future work could incorporate model interpretation techniques to investigate which regions of the heatmaps, and therefore which player combinations, have the greatest influence on the model's predictions.
