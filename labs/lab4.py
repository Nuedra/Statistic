import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

np.random.seed(42)
n = 100
alpha = 0.05

X = stats.norm.rvs(loc=1, scale=np.sqrt(2), size=n)
Y = stats.uniform.rvs(loc=-1, scale=2, size=n)


# 1. Генерация
x_bar = np.mean(X)
y_bar = np.mean(Y)

s2_x = np.var(X, ddof=1)
s2_y = np.var(Y, ddof=1)

r_xy, _   = stats.pearsonr(X, Y)
rho_xy, _ = stats.spearmanr(X, Y)
tau_xy, _ = stats.kendalltau(X, Y)

cols = [
    "Среднее, x̄_i",
    "Оценка дисперсии, s_i^2",
    "КК по Пирсону, r_XY",
    "КК по Спирмену, ρ_XY",
    "КК по Кендаллу, τ_XY",
]

table = pd.DataFrame(index=["X", "Y"], columns=cols, dtype=float)

table.loc["X", "Среднее, x̄_i"] = x_bar
table.loc["X", "Оценка дисперсии, s_i^2"] = s2_x
table.loc["Y", "Среднее, x̄_i"] = y_bar
table.loc["Y", "Оценка дисперсии, s_i^2"] = s2_y

table.loc["X", "КК по Пирсону, r_XY"] = r_xy
table.loc["X", "КК по Спирмену, ρ_XY"] = rho_xy
table.loc["X", "КК по Кендаллу, τ_XY"] = tau_xy

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.expand_frame_repr", False)

print(table.to_string(float_format=lambda v: f"{v:.6f}" if pd.notna(v) else ""))

# Проверка гипотез
r_xy, p_r = stats.pearsonr(X, Y)
rho_xy, p_rho = stats.spearmanr(X, Y)
tau_xy, p_tau = stats.kendalltau(X, Y)

hyp = [r"$H_0: r_{XY}=0$", r"$H_0: \rho_{XY}=0$", r"$H_0: \tau_{XY}=0$"]
pvals = [p_r, p_rho, p_tau]


table2 = pd.DataFrame({
    "Статистическая гипотеза, H0": hyp,
    "p-value": pvals,
})

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.expand_frame_repr", False)

print(table2.to_string(index=False, float_format=lambda v: f"{v:.6g}"))

#2 график
plt.figure()
plt.scatter(X, Y)
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.show()

#3

n = len(X)
k = 5
m = 5

# Группировка
H, x_edges, y_edges, _ = plt.hist2d(X, Y, bins=[k, m])
plt.close()

H = H.astype(int)
row_sum = H.sum(axis=1)
col_sum = H.sum(axis=0)

E = np.outer(row_sum, col_sum) / n

row_labels = [
    f"Δ{i+1} = [{x_edges[i]:.2f}; {x_edges[i+1]:.2f}{')' if i < k-1 else ']'}"
    for i in range(k)
]
col_labels = [
    f"[{y_edges[j]:.2f}; {y_edges[j+1]:.2f}{')' if j < m-1 else ']'}"
    for j in range(m)
]

table = pd.DataFrame(index=row_labels, columns=col_labels, dtype=object)
for i in range(k):
    for j in range(m):
        table.iloc[i, j] = f"{H[i, j]}\n{E[i, j]:.2f}"

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)
pd.set_option("display.expand_frame_repr", False)

print(table.to_string())

k = 5
m = 5

H, _, _, _ = plt.hist2d(X, Y, bins=[k, m])
plt.close()
H = H.astype(int)

chi2, p_value, dof, expected = stats.chi2_contingency(H)

print(f"Выборочное значение статистики: {chi2:.6g}")
print(f"p-value: {p_value:.6g}")

#4

lambdas = np.linspace(0, 1, 10000)

r_XU = []
rho_XU = []
tau_XU = []

for l in lambdas:
    U = l * X + (1 - l) * Y
    r_XU.append(stats.pearsonr(X, U)[0])
    rho_XU.append(stats.spearmanr(X, U)[0])
    tau_XU.append(stats.kendalltau(X, U)[0])

plt.figure()
plt.plot(lambdas, r_XU, label=r"$r_{XU}$")
plt.plot(lambdas, rho_XU, label=r"$\rho_{XU}$")
plt.plot(lambdas, tau_XU, label=r"$\tau_{XU}$")
plt.xlabel(r"$\lambda$")
plt.ylabel(r"$r_{XU}, \rho_{XU}, \tau_{XU}$")
plt.title(r"Графики зависимостей $r_{XU}(\lambda)$, $\rho_{XU}(\lambda)$ и $\tau_{XU}(\lambda)$")
plt.legend()
plt.grid(True)
plt.show()

r_XV = []
rho_XV = []
tau_XV = []

for l in lambdas:
    V = l * (X**3) + (1 - l) * (Y**3)
    r_XV.append(stats.pearsonr(X, V)[0])
    rho_XV.append(stats.spearmanr(X, V)[0])
    tau_XV.append(stats.kendalltau(X, V)[0])

plt.figure()
plt.plot(lambdas, r_XV, label=r"$r_{XV}$")
plt.plot(lambdas, rho_XV, label=r"$\rho_{XV}$")
plt.plot(lambdas, tau_XV, label=r"$\tau_{XV}$")
plt.xlabel(r"$\lambda$")
plt.ylabel(r"$r_{XV}, \rho_{XV}, \tau_{XV}$")
plt.title(r"Графики зависимостей $r_{XV}(\lambda)$, $\rho_{XV}(\lambda)$ и $\tau_{XV}(\lambda)$")
plt.legend()
plt.grid(True)
plt.show()

fig, axes = plt.subplots(2, 2, figsize=(10, 6))

title_fs = 9  # размер шрифта заголовков

axes[0, 0].scatter(X, Y**3)
axes[0, 0].set_xlabel(r"$X$")
axes[0, 0].set_ylabel(r"$V$")
axes[0, 0].set_title(r"Диаграмма рассеяния случайных величин $X$ и $V$ при $\lambda = 0$", fontsize=title_fs)

axes[0, 1].scatter(stats.rankdata(X), stats.rankdata(Y**3))
axes[0, 1].set_xlabel(r"$R$")
axes[0, 1].set_ylabel(r"$S$")
axes[0, 1].set_title(r"Диаграмма рассеяния рангов случайных величин $X$ и $V$ при $\lambda = 0$", fontsize=title_fs)

axes[1, 0].scatter(X, X**3)
axes[1, 0].set_xlabel(r"$X$")
axes[1, 0].set_ylabel(r"$V$")
axes[1, 0].set_title(r"Диаграмма рассеяния случайных величин $X$ и $V$ при $\lambda = 1$", fontsize=title_fs)

axes[1, 1].scatter(stats.rankdata(X), stats.rankdata(X**3))
axes[1, 1].set_xlabel(r"$R$")
axes[1, 1].set_ylabel(r"$S$")
axes[1, 1].set_title(r"Диаграмма рассеяния рангов случайных величин $X$ и $V$ при $\lambda = 1$", fontsize=title_fs)

plt.tight_layout()
plt.show()









