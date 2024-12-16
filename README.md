# Water Quality Data Analysis and Visualization in Australia

## Data:
Data is available on Kaggle [Real Time Water Quality Data](https://www.kaggle.com/datasets/ivivan/real-time-water-quality-data) and [Australian Government](https://data.gov.au/data/) serves for analysis and visualization.


## Model
The time series imputation model utilizes neural networks, as referenced in the paper [A dual-head attention model for time series data imputation](https://www.sciencedirect.com/science/article/pii/S016816992100394X)


## 🌟 **Project Introduction**
This project focuses on analyzing, visualizing, and imputing time series data for water quality in the Mulgrave River at Deeral, Queensland, Australia. We applied advanced techniques in data processing and deep learning to deliver practical and valuable outcomes.

### **Project Highlights:**
- Analyzed and visualized water quality data with multiple influencing factors such as water flow, water level, turbidity, and water temperature.
- Collected and integrated supplementary data from related sources to extend the analysis.
- Developed a time series data imputation model based on bidirectional Gated Recurrent Units (GRUs), achieving superior performance compared to traditional methods like KNN.

---

## 🔑 **Key Achievements**

### 1. **Data Analysis and Visualization**
- Utilized detailed charts (boxplots, heatmaps, scatterplots) to illustrate the distribution and correlation among factors, such as:
  - Rainfall significantly impacting turbidity and water flow.
  - Water temperature varying seasonally and closely linked to air temperature.

### 2. **Handling Missing Data**
- Analyzed and processed datasets with over 65% missing values in certain attributes.
- Applied bidirectional GRU-based deep learning techniques to impute continuous missing data, improving accuracy by **14 times** compared to KNN in experiments.

### 3. **Model Results**
- The imputation model achieved:
  - RMSE: **0.032**
  - MAE: **0.029**
  - DTW: **0.071** (with a gap of 6 consecutive missing values).
- Provided a scalable approach applicable to other time-series data projects.

---

## 🛠️ **Technologies Used**
- **Programming Language**: Python.
- **Libraries**:
  - Data processing and analysis: Pandas, NumPy.
  - Visualization: Matplotlib, Seaborn.
  - Deep Learning: PyTorch.
- **Additional Tools**:
  - Jupyter Notebook.
  - Tableau: For interactive dashboards.

---

## 📊 **Real-World Applications**
1. **Water Quality Management:**
   - Provides detailed insights to help managers better understand factors affecting water quality.
   - Issues early warnings on unusual environmental data fluctuations.
2. **Time Series Data Modeling:**
   - The imputation solution can be extended to fields like finance, healthcare, and IoT.

---

## 🚀 **Future Developments**
- Optimize the GRU model to handle larger data gaps.
- Integrate advanced models like Transformers for time-series forecasting.
- Build an automated system for real-time data collection and analysis.

---

*This is the result of a course project - a collaborative effort by the team members.*
