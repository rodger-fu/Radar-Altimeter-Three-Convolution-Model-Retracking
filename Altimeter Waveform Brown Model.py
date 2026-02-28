import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve, windows
from scipy.special import i0, erf
import matplotlib
from scipy.interpolate import interp1d

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# ============================================================================
# 1. 参数设置 - 修正时间轴参考
# ============================================================================
# 雷达系统参数
c = 3e8  # 光速 [m/s]
f0 = 13.5e9  # 载频 [Hz]
lambda_radar = c / f0  # 波长 [m]
Bandwidth = 320e6  # 带宽 [Hz]
T_pulse = 50e-6  # 脉冲宽度 [s]
Fs = 2 * Bandwidth  # 采样率 [Hz]
h = 800e3  # 卫星高度 [m]
G0 = 10 ** (40 / 10)  # 天线增益
theta_w = np.deg2rad(1.2)  # 天线波束宽度 [rad]

# 海况参数
SWH = 2.0  # 有效波高 [m]
sigma_s = SWH / 4 * (2 / c)  # 转换为时间域的标准差 [s] 海面高度的均方根
lambda_s = -0.1  # 海面偏度
kappa_s = 0.1  # 海面峰度

# 几何参数
xi = np.deg2rad(0.2)  # 离天底角 [rad]

# 修正：以星下点回波时延为时间轴参考零点
t_delay_nadir = 2 * h / c  # 星下点回波时延 [s] - 作为时间参考零点
print(f"星下点回波时延 (时间参考零点): {t_delay_nadir * 1e6:.2f} μs")

# 考虑指向角影响的回波时延 - 这是相对于星下点的延迟
rho_min = h * np.tan(xi)  # 星下点到波束中心交点的地面距离
r_min = np.sqrt(h ** 2 + rho_min ** 2)  # 最短斜距
t_delay_min = 2 * r_min / c  # 最短回波时延
pointing_delay = t_delay_min - t_delay_nadir  # 指向角引起的额外时延
print(f"指向角引起的额外时延: {pointing_delay * 1e9:.1f} ns")

# 时间轴设置 - 以星下点回波时延为参考零点
t_span = 300e-9  # 总时间跨度 [s]
Nt = 2048
t_center = 0  # 以星下点回波时延为参考零点
t_start = t_center - t_span / 2
t_end = t_center + t_span / 2
t = np.linspace(t_start, t_end, Nt)  # 时间轴 [s]，零点对应星下点回波
dt = t[1] - t[0]

print(f"时间轴范围: {t_start * 1e9:.1f} ~ {t_end * 1e9:.1f} ns")
print(f"时间轴中心 (星下点回波): 0 ns")

# ============================================================================
# 2. 计算平均平坦海面冲激响应 P_FS(t) - 修正时间参考
# ============================================================================
# 天线波束宽度参数 gamma
gamma = 2 * np.log(4) / (np.sin(theta_w / 2) ** 2)

# P_FS(t) 的参数
delta = (4 / gamma) * (c / h) * np.cos(2 * xi)
beta = (4 / gamma) * np.sqrt(c / h) * np.sin(2 * xi)
A = 1.0  # 幅度因子

# 计算 P_FS(t) - 修正时间参考
P_FS = np.zeros_like(t)

# 对于每个时间t，计算对应的地面距离ρ
# 实际回波时间 = t + t_delay_nadir (因为t以星下点回波为参考零点)
r_t = c * (t + t_delay_nadir) / 2  # 实际斜距

# 只处理有效的斜距 (r_t >= h)
valid_mask = r_t >= h
rho_t = np.zeros_like(t)
rho_t[valid_mask] = np.sqrt(r_t[valid_mask] ** 2 - h ** 2)

# 计算对应的离轴角θ和入射角ψ
theta_t = np.zeros_like(t)
psi_t = np.zeros_like(t)

# 只在有效区域计算角度
theta_t[valid_mask] = np.arctan(rho_t[valid_mask] / h) - xi
psi_t[valid_mask] = np.arctan(rho_t[valid_mask] / h)

# 使用更精确的公式计算P_FS
P_FS_temp = np.zeros_like(t)

if np.any(valid_mask):
    # 天线增益 (高斯近似)
    G_sq = np.exp(-(4 / gamma) * np.sin(theta_t[valid_mask]) ** 2)

    # 距离衰减因子 (1/r^4)
    range_factor = 1 / r_t[valid_mask] ** 4

    # 面积元 dA = ρ dρ dφ，但dρ/dt需要转换
    # dt/dρ = (2/c) * (ρ/r) => dρ/dt = (c/2) * (r/ρ)
    jacobian = np.zeros_like(rho_t[valid_mask])
    nonzero_rho_mask = rho_t[valid_mask] > 1e-6  # 避免除以零
    jacobian[nonzero_rho_mask] = (c / 2) * (r_t[valid_mask][nonzero_rho_mask] / rho_t[valid_mask][nonzero_rho_mask])

    P_FS_temp[valid_mask] = A * G_sq * range_factor * jacobian

# 归一化
if np.max(P_FS_temp) > 0:
    P_FS = P_FS_temp / np.max(P_FS_temp)


# ============================================================================
# 3. 生成海面高程分布函数 q_s(t) - 修正时间参考
# ============================================================================
def hermite_poly(n, z):
    if n == 3:
        return z ** 3 - 3 * z
    elif n == 4:
        return z ** 4 - 6 * z ** 2 + 3
    elif n == 6:
        return z ** 6 - 15 * z ** 4 + 45 * z ** 2 - 15
    else:
        raise ValueError("只实现了 H3, H4, H6")


# q_s(t)的时间参考也是以星下点回波为参考零点
# 海面高程变化引起的时延变化直接体现在t上
z_t = t / sigma_s  # 以星下点回波为参考零点

gaussian = np.exp(-0.5 * z_t ** 2) / (np.sqrt(2 * np.pi) * sigma_s)
H3 = hermite_poly(3, z_t)
H4 = hermite_poly(4, z_t)
H6 = hermite_poly(6, z_t)

q_s = gaussian * (1 + (lambda_s / 6) * H3 + (kappa_s / 24) * H4 + (lambda_s ** 2 / 72) * H6)
q_s_integral = np.trapz(q_s, t)
if q_s_integral > 0:
    q_s = q_s / q_s_integral

# ============================================================================
# 4. 生成雷达系统点目标响应 s_r(t) - 修正时间参考
# ============================================================================
# 系统响应应该出现在指向点回波时延处
# 指向点相对于星下点的时延 = pointing_delay
sigma_r = 1 / (2 * Bandwidth)  # 理论分辨率对应的时宽
print(f"系统响应标准差: {sigma_r * 1e9:.2f} ns")

# 系统响应出现在指向点回波时延处（相对于星下点）
s_r = np.exp(-0.5 * ((t - pointing_delay) / sigma_r) ** 2)

# 归一化面积
s_r_integral = np.trapz(s_r, t)
if s_r_integral > 0:
    s_r = s_r / s_r_integral

# ============================================================================
# 5. 进行三项卷积
# ============================================================================
P_FS = np.nan_to_num(P_FS)
q_s = np.nan_to_num(q_s)
s_r = np.nan_to_num(s_r)

print(f"卷积前检查 - P_FS最大值: {np.max(P_FS):.6f}, 位置: {t[np.argmax(P_FS)]*1e9:.1f} ns")
print(f"卷积前检查 - q_s最大值: {np.max(q_s):.6f}, 位置: {t[np.argmax(q_s)]*1e9:.1f} ns")
print(f"卷积前检查 - s_r最大值: {np.max(s_r):.6f}, 位置: {t[np.argmax(s_r)]*1e9:.1f} ns")

# 检查是否有非零值
if np.max(P_FS) > 0 and np.max(q_s) > 0 and np.max(s_r) > 0:
    W1 = convolve(P_FS, q_s, mode='same') * dt
    W_final = convolve(W1, s_r, mode='same') * dt

    if np.max(W_final) > 0:
        W_final = W_final / np.max(W_final)
    else:
        print("警告: 最终卷积结果全为零")
        W_final = np.zeros_like(t)
else:
    print("错误: 卷积分量中存在全零数组")
    W_final = np.zeros_like(t)

# ============================================================================
# 6. 可视化结果
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 时间显示 (以星下点回波为参考零点)
t_ns = t * 1e9

# 6.1 三个卷积分量
ax = axes[0, 0]
ax.plot(t_ns, P_FS, 'b-', linewidth=2, label=r'$P_{FS}(t)$')
ax.plot(t_ns, q_s / np.max(q_s) if np.max(q_s) > 0 else q_s, 'g-', linewidth=2, label=r'$q_s(t)$')
ax.plot(t_ns, s_r / np.max(s_r) if np.max(s_r) > 0 else s_r, 'r-', linewidth=2, label=r'$s_r(t)$')
ax.axvline(0, color='k', linestyle='--', alpha=0.5, label='星下点回波')
ax.axvline(pointing_delay * 1e9, color='r', linestyle='--', alpha=0.5, label='指向点回波')
ax.set_xlabel('时间 [ns] (星下点回波=0)')
ax.set_ylabel('归一化幅度')
ax.set_title('三项卷积模型的三个分量')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim([-150, 150])

# 6.2 中间卷积结果
ax = axes[0, 1]
if np.max(W1) > 0:
    W1_plot = W1 / np.max(W1)
else:
    W1_plot = W1
ax.plot(t_ns, W1_plot, 'm-', linewidth=2, label=r'$P_{FS}(t) * q_s(t)$')
ax.axvline(0, color='k', linestyle='--', alpha=0.5, label='星下点回波')
ax.axvline(pointing_delay * 1e9, color='r', linestyle='--', alpha=0.5, label='指向点回波')
ax.set_xlabel('时间 [ns] (星下点回波=0)')
ax.set_ylabel('归一化幅度')
ax.set_title('前两项卷积结果')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim([-150, 150])

# 6.3 系统响应细节
ax = axes[1, 0]
ax.plot(t_ns, s_r / np.max(s_r) if np.max(s_r) > 0 else s_r, 'r-', linewidth=2, label='系统响应')
ax.axvline(0, color='k', linestyle='--', alpha=0.5, label='星下点回波')
ax.axvline(pointing_delay * 1e9, color='r', linestyle='--', alpha=0.5, label='指向点回波')
ax.set_xlabel('时间 [ns] (星下点回波=0)')
ax.set_ylabel('归一化幅度')
ax.set_title('系统点目标响应 (高斯近似)')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim([-50, 100])

# 6.4 最终的平均回波波形
ax = axes[1, 1]
ax.plot(t_ns, W_final, 'k-', linewidth=2, label='模拟回波波形')
ax.axvline(0, color='k', linestyle='--', alpha=0.5, label='星下点回波')
ax.axvline(pointing_delay * 1e9, color='r', linestyle='--', alpha=0.5, label='指向点回波')
ax.set_xlabel('时间 [ns] (星下点回波=0)')
ax.set_ylabel('归一化功率')
ax.set_title(f'最终平均回波波形 (SWH={SWH}m, ξ={np.rad2deg(xi):.1f}°)')
ax.legend()
ax.grid(True, alpha=0.3)
# ax.set_xlim([-150, 150])

plt.tight_layout()
plt.show()

# ============================================================================
# 7. 输出关键参数
# ============================================================================
print("=" * 50)
print("修正后仿真参数总结")
print("=" * 50)
print(f"几何关系:")
print(f"  卫星高度: {h / 1e3:.0f} km")
print(f"  离天底角: {np.rad2deg(xi):.2f} °")
print(f"  星下点回波时延: {t_delay_nadir * 1e6:.2f} μs")
print(f"  指向点回波时延: {t_delay_min * 1e6:.2f} μs")
print(f"  指向角引起的额外时延: {pointing_delay * 1e9:.1f} ns")
print(f"雷达系统:")
print(f"  带宽: {Bandwidth / 1e6:.0f} MHz")
print(f"  距离分辨率: {c / (2 * Bandwidth):.2f} m")
print(f"  系统响应宽度 (σ_r): {sigma_r * 1e9:.2f} ns")
print(f"海况:")
print(f"  有效波高 SWH: {SWH} m")
print(f"  海面高程标准差 (σ_s): {sigma_s * 1e9:.2f} ns")
print(f"波形特征:")
print(f"  P_FS有效点数: {np.sum(valid_mask)}/{Nt}")
print(f"  P_FS峰值时间: {t_ns[np.argmax(P_FS)]:.1f} ns")
print(f"  q_s峰值时间: {t_ns[np.argmax(q_s)]:.1f} ns")
print(f"  s_r峰值时间: {t_ns[np.argmax(s_r)]:.1f} ns")
print(f"  最终波形峰值时间: {t_ns[np.argmax(W_final)]:.1f} ns")
print("=" * 50)