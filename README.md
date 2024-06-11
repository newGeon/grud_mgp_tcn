# GUR-D & MGP-TCN for Sepsis Prediction on the MIMIC-III Dataset
MIMIC-III datatabse + GUR-D &amp; MGP-TCN

## Code Review
- https://github.com/BorgwardtLab/mgp-tcn
- 위의 Gitub 코드를 기반으로 Data Augmentaion 부분 알고리즘(GUR-D) 추가 (한글)
- Data Augmentation partial algorithm (GRU-D) added based on the above Github code (English)

## Code Execution
### 1. preprocessing
- path : ~/grud_mgp_tcn/preprocessing/
- Origin preprocessing code start
```
python3 deta_execution_v2_0_1.py
```

- GRU-D preprocessing code start
```
python3 deta_execution_v2_1_1.py
```
- Linear Regression preprocessing code start
```
python3 deta_execution_v2_2_1.py
```
- Clinical preprocessing code start
```
python3 deta_execution_v2_3_1.py
```
