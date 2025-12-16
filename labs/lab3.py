import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

#генерация выборок
alpha = 0.05
seed = 42
rng = np.random.default_rng(seed)

X1 = stats.norm.rvs(loc=1, scale=math.sqrt(2), size=100, random_state=rng)     # N(1,2)
X2 = stats.uniform.rvs(loc=0, scale=2, size=150, random_state=rng)             # R(0,2)
X3 = stats.norm.rvs(loc=1, scale=math.sqrt(3), size=200, random_state=rng)     # N(1,3)

def sample_stats(x):
    xbar = float(np.mean(x))                 # \bar{x}
    S2 = float(np.var(x, ddof=1))            # S^2 (несмещённая)
    S = float(np.sqrt(S2))                   # S
    return xbar, S2, S

x1bar, S1_2, S1 = sample_stats(X1)
x2bar, S2_2, S2 = sample_stats(X2)
x3bar, S3_2, S3 = sample_stats(X3)

Sp2 = ((len(X1)-1)*S1_2 + (len(X2)-1)*S2_2 + (len(X3)-1)*S3_2) / ((len(X1)-1)+(len(X2)-1)+(len(X3)-1))
Sp = math.sqrt(Sp2)

print("Выборочные характеристики (х, s^2, s):")
print(f"X1: x̄1 = {x1bar:.6f},  s1^2 = {S1_2:.6f},  s1 = {S1:.6f},  n1 = {len(X1)}")
print(f"X2: x̄2 = {x2bar:.6f},  s2^2 = {S2_2:.6f},  s2 = {S2:.6f},  n2 = {len(X2)}")
print(f"X3: x̄3 = {x3bar:.6f},  s3^2 = {S3_2:.6f},  s3 = {S3:.6f},  n3 = {len(X3)}")


xbar_all = (len(X1)*x1bar + len(X2)*x2bar + len(X3)*x3bar) / (len(X1)+len(X2)+len(X3))
print("\nPooled (сводная оценка дисперсии внутри групп):")
print(f"x = {xbar_all:.6f}, s_p^2 = {Sp2:.6f},  s_p = {Sp:.6f}")


# 2 Box-and-Whisker
plt.figure()
plt.boxplot([X1, X2, X3], vert=False, tick_labels=['X1', 'X2', 'X3'])
plt.title('Box-and-Whisker')
plt.xlabel('Значения')
plt.ylabel('Выборки')
plt.tight_layout()
plt.show()


# 3 Проверка равенства дисперсий (критерий Бартлетта)
bart_stat, bart_p = stats.bartlett(X1, X2, X3)

print('\n--- 3. Критерий Бартлетта ---')
print(f'chi2_B (статистика) = {bart_stat:.6g}')
print(f'p-value = {bart_p:.6g}')

#  4 Однофакторный дисперсионный анализ
F_stat, p_value = stats.f_oneway(X1, X2, X3)
anova_decision = 'Отвергнуть H0' if p_value < alpha else 'Не отвергать H0'


groups = [X1, X2, X3]
ns = [len(g) for g in groups]
means = [np.mean(g) for g in groups]
grand = np.mean(np.concatenate(groups))

Q_A = sum(n*(m-grand)**2 for n, m in zip(ns, means))                # межгрупповая
Q_E = sum(((g - m)**2).sum() for g, m in zip(groups, means))        # внутригрупповая
Q = ((np.concatenate(groups) - grand)**2).sum()                     # общая

v1 = len(groups) - 1
v2 = sum(ns) - len(groups)

sA2 = Q_A / v1
sE2 = Q_E / v2

eta2 = Q_A / Q
eta = math.sqrt(eta2)

n = sum(ns)
k = len(groups)

D_between = Q_A / n
D_within  = Q_E / n
D_total   = Q   / n

s2_within  = n/(n-k) * D_within
s2_total   = n/(n-1) * D_total
s2_between = n/(k-1) * D_between

print(f"n = {n}, k = {k}")
print(f"D~_межгр = {D_between:.6g},   df = k-1 = {k-1},   n/(k-1)*D~_межгр = {s2_between:.6g}")
print(f"D~_внутр = {D_within:.6g},    df = n-k = {n-k},   n/(n-k)*D~_внутр = {s2_within:.6g}")
print(f"D~_общ   = {D_total:.6g},     df = n-1 = {n-1},   n/(n-1)*D~_общ   = {s2_total:.6g}")
print("\nЭмпирический коэффициент детерминации:")
print(f"eta^2 = {eta2:.6g}")

print("\nЭмпирическое корреляционное отношение:")
print(f"eta = {eta:.6g}")


X = [X1, X2, X3]
n = [len(x) for x in X]
k = len(X)


#  5 Метод линейных контрастов: доверительные интервалы для m1,...,mk
intervals = pd.DataFrame(index=["Нижняя граница", "Верхняя граница"])

for data, N, i in zip(X, n, range(1, k + 1)):
    std_err = stats.sem(data)
    low, high = stats.t.interval(1 - alpha, df=N - 1, loc=np.mean(data), scale=std_err)
    intervals[f"X{i}"] = [low, high]

print("\nДоверительные интервалы для математических ожиданий m_i:")
print(intervals)

plt.figure()
for interval, y in zip(intervals.columns, range(1, k + 1)):
    center = intervals[interval].mean()
    half = center - intervals[interval].iloc[0]
    plt.errorbar(center, y, xerr=half, fmt="o", label=f"m{y}")

plt.yticks(range(1, k + 1), [f"X{i}" for i in range(1, k + 1)])
plt.legend(loc="upper left")
plt.title("Доверительные интервалы для математических ожиданий")
plt.tight_layout()
plt.show()

# 6 Попарные сравнения
X_pooled = np.concatenate(X)
grouped = pd.DataFrame({"data": X_pooled, "group": np.repeat(range(1, k + 1), n)})

print("\nПервые строки таблицы pooled-данных:")
print(grouped.head())

result = pairwise_tukeyhsd(endog=grouped["data"], groups=grouped["group"], alpha=alpha)

print("\nРезультат Tukey HSD (summary):")
print(result)



