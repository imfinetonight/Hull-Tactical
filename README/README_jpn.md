# Hull Tactical - Market Prediction
### 🥈 Silver Medal Solution ###
KaggleのHull Tactical Market Predictionコンペティションにおいて、3,677チーム中174位にランクインしました。  

![Pipeline](../images/174th.png)
(※現ユーザー名: **Water Baby**)  

## 概要
このリポジトリでは、**Kaggle Hull Tactical Market Predictionコンペティション**　における私の解法を掲示します。  
本コンペティションの課題は、ボラティリティの制約を考慮しつつ、特定の市場データの特徴量を用いて、S&P500の超過収益率で表される株式市場の収益率を予測することです。  
推論はレイテンシ制約下でサーバーゲートウェイを介して実行されました。  
*コンペティションページ：* https://www.kaggle.com/competitions/hull-tactical-market-prediction  

本リポジトリでは、特徴量エンジニアリング、アンサンブル学習、市場レジームの検出、および適応型ポジションサイジングを網羅したエンドツーエンドのクオンツトレーディング・パイプラインを提示します。  
このソリューションでは、アンサンブル予測エンジンと適応型ポジションサイジングを組み合わせることで、予測とポジショニングを分離しています。

## アーキテクチャ
このソリューションは、2つの主要なコンポーネントで構成されています。

1. 予測エンジン
   - Feature engineering
   - Model ensemble (CatBoost + LightGBM)
   - Ridge stacking

2. ポジショニングエンジン
   - Market regime detection
   - Adaptive risk adjustment
   - Dynamic position sizing

![Pipeline](../images/Architecture.png)  

## 主なハイライト
- Adaptive position sizing
- Market regime detection
- Ridge stacking (CatBoost × LightGBM)
- Stateful inference pipeline
- Time-series feature engineering

## 特徴量エンジニアリング
時系列および市場の特性を考慮した特徴量をいくつか実装しました。
また、すべての前処理は、再利用可能な scikit-learn の Pipeline にカプセル化されました。

- Lag features
- Rolling statistics
- Momentum
- Market regime features
- Rank-transformed anonymous features
- PCA-compressed latent factors
- Seasonal features

## モデリング
Ensemble Models:

- CatBoost (4 PCA variants)
- LightGBM (1 PCA variant)
- Ridge stacking

Validation:

- Time Series Cross Validation

## 学び
How to:

- Build reproducible ML pipelines
- Prevent time-series leakage
- Design features for online inference
- Separate forecasting from execution
- Transform model predictions into adaptive trading positions
- Optimize latency-aware inference pipelines

正確な予測を生成することは、トレーディングシステムの一側面に過ぎず、予測を堅牢なポジションへと変換することも、同様に重要であることを学びました。

## Repository Structure

```
Hull_Tactical/
│── README/
│   ├── README.md
│   └── README_jpn.md
│
│── images/
│   ├── 174th.png
│   └── Architecture.png
│
│── notebooks/
│   ├── inference.ipynb
│   └── train.ipynb
│
│── src/
│   ├── inference.py
│   └── train.py
│
│── .gitignore
│── LICENSE
└── requirements.txt
```

## 環境設定

本プロジェクトは完全にローカライズおよび最適化されており、Appleシリコン（M1/M2/M3）環境を含む macOS 上で、一般的なコンパイルの問題に遭遇することなく安定して動作します。

### 仮想環境
依存関係の競合を避けるため、`conda`を使用して隔離された環境を構築することを推奨します。 

```bash
# Create and activate a Python 3.10 environment
conda create -n hull python=3.10 -y
conda activate hull
```

### インストール
```bash
pip install -r requirements.txt
```
