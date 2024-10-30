# Lake Mendota Ice Coverage Analysis

This project applies linear regression to predict the ice coverage days on Lake Mendota, using historical data to model trends and make future predictions. The analysis includes data curation, visualization, normalization, and optimization with gradient descent.

---

## 📑 Project Overview

Lake Mendota, Wisconsin, has long been monitored for annual ice coverage, providing insights into climate patterns over time. This project uses data from 1855 to 2023 to model the relationship between the year and the number of days the lake is frozen, helping to predict future ice coverage trends.

## 🎯 Objectives

1. **Data Curation**: Clean and structure raw ice coverage data from Lake Mendota.
2. **Visualization**: Plot historical data to identify trends in ice coverage days.
3. **Linear Regression**: Use a closed-form solution and gradient descent to fit a model predicting ice coverage days.
4. **Prediction**: Forecast ice coverage for the 2023-24 season.
5. **Model Interpretation**: Analyze the model’s weight and interpret its implications for ice coverage.
6. **Limitations**: Explore the limitations and potential inaccuracies of the model.

---

## 📁 Project Structure

### Files
- **`ice_days.csv`**: Curated dataset containing the years and corresponding ice coverage days.
- **`analysis.py`**: Python script to read data, run linear regression, and perform predictions.
- **`data_plot.jpg`**: Visualization of the number of ice coverage days over the years.
- **`loss_plot.jpg`**: Loss plot showing the error reduction over gradient descent iterations.

---

### 🚀 Running the Code

**Dependencies**: Make sure you have the following libraries installed: `numpy`, `matplotlib`, `pandas`.

To run the analysis, use the following command:

```bash
python analysis.py ice_days.csv <learning_rate> <iterations>
```

Where:
- `<learning_rate>` is a float, specifying the learning rate for gradient descent (e.g., `0.01`).
- `<iterations>` is an integer, specifying the number of iterations for gradient descent (e.g., `500`).

---

## 📊 Output Files
- **`data_plot.jpg`**: Visual representation of ice coverage days over time.
- **`loss_plot.jpg`**: Loss plot for gradient descent showing convergence over time.

---

## 📈 Results and Observations

The analysis indicates a predicted **85.5 ice coverage days** for the 2023-24 winter season on Lake Mendota. 

This prediction is derived from a linear regression model based on historical data, which assumes consistent trends over time. However, external factors such as climate change and its influence on weather patterns may cause significant deviations from this estimate. These factors can introduce variability that a simple linear model may not capture, suggesting the need for more complex models to improve future predictions.

