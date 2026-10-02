"""
선형회귀와 비선형회귀 프로그래밍 실습 (데이터처리개론)

- 데이터: UCI Auto MPG (공개 데이터)
  자동차의 마력(horsepower)으로 연비(mpg)를 예측한다.
- 선형회귀:   y = w*x + b
- 비선형회귀: y = w2*x^2 + w1*x + b  (2차 다항 회귀)
- 두 모델 모두 numpy만 써서 '경사하강법'으로 직접 학습한다.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1. 데이터 불러오기
# ------------------------------------------------------------
UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
MIRROR_URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv"


def load_data():
    """Auto MPG 데이터를 불러와 (마력, 연비) 배열로 돌려준다."""
    try:
        cols = ["mpg", "cylinders", "displacement", "horsepower",
                "weight", "acceleration", "model_year", "origin"]
        df = pd.read_csv(UCI_URL, names=cols, na_values="?",
                         comment="\t", sep=" ", skipinitialspace=True)
        print("UCI 저장소에서 데이터를 불러왔습니다.")
    except Exception:
        df = pd.read_csv(MIRROR_URL)
        print("미러 사이트에서 데이터를 불러왔습니다.")

    df = df[["horsepower", "mpg"]].dropna()      # 결측치 제거
    x = df["horsepower"].to_numpy(dtype=float)
    y = df["mpg"].to_numpy(dtype=float)
    return x, y


# ------------------------------------------------------------
# 2. 학습/테스트 데이터 나누기 (80% / 20%)
# ------------------------------------------------------------
def train_test_split(x, y, test_ratio=0.2, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(x))
    n_test = int(len(x) * test_ratio)
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return x[train_idx], x[test_idx], y[train_idx], y[test_idx]


# ------------------------------------------------------------
# 3. 평가 지표
# ------------------------------------------------------------
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot


# ------------------------------------------------------------
# 4. 선형회귀 (경사하강법)
# ------------------------------------------------------------
class LinearRegressionGD:
    """y = w*x + b 를 경사하강법으로 학습"""

    def __init__(self, lr=0.1, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.w = 0.0
        self.b = 0.0
        self.loss_history = []

    def predict(self, x):
        return self.w * x + self.b

    def fit(self, x, y):
        n = len(x)
        for _ in range(self.epochs):
            y_pred = self.predict(x)
            error = y_pred - y
            # 손실 L = (1/n) * Σ(y_pred - y)^2 의 편미분
            dw = (2 / n) * np.sum(error * x)
            db = (2 / n) * np.sum(error)
            # 기울기 반대 방향으로 조금씩 이동
            self.w -= self.lr * dw
            self.b -= self.lr * db
            self.loss_history.append(mse(y, y_pred))
        return self


# ------------------------------------------------------------
# 5. 비선형회귀: 2차 다항 회귀 (경사하강법)
# ------------------------------------------------------------
class PolynomialRegressionGD:
    """y = w2*x^2 + w1*x + b 를 경사하강법으로 학습"""

    def __init__(self, lr=0.05, epochs=3000):
        self.lr = lr
        self.epochs = epochs
        self.w1 = 0.0
        self.w2 = 0.0
        self.b = 0.0
        self.loss_history = []

    def predict(self, x):
        return self.w2 * x ** 2 + self.w1 * x + self.b

    def fit(self, x, y):
        n = len(x)
        for _ in range(self.epochs):
            y_pred = self.predict(x)
            error = y_pred - y
            dw2 = (2 / n) * np.sum(error * x ** 2)
            dw1 = (2 / n) * np.sum(error * x)
            db = (2 / n) * np.sum(error)
            self.w2 -= self.lr * dw2
            self.w1 -= self.lr * dw1
            self.b -= self.lr * db
            self.loss_history.append(mse(y, y_pred))
        return self


# ------------------------------------------------------------
# 6. 메인 실행
# ------------------------------------------------------------
def main():
    x, y = load_data()
    print(f"데이터 개수: {len(x)}개")

    x_train, x_test, y_train, y_test = train_test_split(x, y)

    # 표준화: 마력 값(46~230)이 커서 그대로 쓰면 경사하강법이 발산하기 쉽다.
    #         학습 데이터의 평균/표준편차로 (x - 평균) / 표준편차 변환
    mean, std = x_train.mean(), x_train.std()
    xs_train = (x_train - mean) / std
    xs_test = (x_test - mean) / std

    # 학습
    lin = LinearRegressionGD().fit(xs_train, y_train)
    poly = PolynomialRegressionGD().fit(xs_train, y_train)

    # 평가
    results = {}
    for name, model in [("선형회귀", lin), ("비선형회귀(2차)", poly)]:
        pred = model.predict(xs_test)
        results[name] = (mse(y_test, pred), r2_score(y_test, pred))

    print("\n===== 학습된 식 (x는 표준화된 마력) =====")
    print(f"선형회귀   : y = {lin.w:.3f}x + {lin.b:.3f}")
    print(f"비선형회귀 : y = {poly.w2:.3f}x² + {poly.w1:.3f}x + {poly.b:.3f}")

    print("\n===== 테스트 데이터 성능 =====")
    print(f"{'모델':<14}{'MSE':>10}{'R²':>10}")
    for name, (m, r) in results.items():
        print(f"{name:<14}{m:>10.3f}{r:>10.3f}")

    # 검증: numpy의 최소제곱 해와 비교 (경사하강법이 제대로 수렴했는지 확인)
    print("\n===== 검증 (np.polyfit 정답과 비교) =====")
    print("선형  :", np.round(np.polyfit(xs_train, y_train, 1), 3))
    print("2차   :", np.round(np.polyfit(xs_train, y_train, 2), 3))

    # 그래프 1: 데이터와 회귀선
    plt.rcParams["axes.unicode_minus"] = False
    grid = np.linspace(x.min(), x.max(), 200)
    grid_s = (grid - mean) / std
    plt.figure(figsize=(8, 5))
    plt.scatter(x_train, y_train, s=12, alpha=0.5, label="train data")
    plt.scatter(x_test, y_test, s=12, alpha=0.7, label="test data")
    plt.plot(grid, lin.predict(grid_s), "r-", lw=2,
             label=f"Linear (R2={results['선형회귀'][1]:.3f})")
    plt.plot(grid, poly.predict(grid_s), "g-", lw=2,
             label=f"Polynomial deg 2 (R2={results['비선형회귀(2차)'][1]:.3f})")
    plt.xlabel("Horsepower")
    plt.ylabel("MPG")
    plt.title("Auto MPG: Linear vs Nonlinear Regression")
    plt.legend()
    plt.tight_layout()
    plt.savefig("regression_result.png", dpi=150)

    # 그래프 2: 손실(MSE) 감소 과정
    plt.figure(figsize=(8, 4))
    plt.plot(lin.loss_history, label="Linear")
    plt.plot(poly.loss_history, label="Polynomial deg 2")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("MSE (log scale)")
    plt.title("Training Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("loss_curve.png", dpi=150)

    print("\n그래프 저장 완료: regression_result.png, loss_curve.png")
    plt.show()


if __name__ == "__main__":
    main()
