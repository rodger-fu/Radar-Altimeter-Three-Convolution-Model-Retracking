import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
from scipy.special import erf
import matplotlib
import time

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# ================= 公共参数设置 =================
c = 3e8  # 光速 [m/s]
f0 = 13.5e9  # 载频 [Hz]
lambda_radar = c / f0
Bandwidth = 320e6  # 带宽 [Hz]
T_pulse = 50e-6  # 脉冲宽度 [s]
Fs = 2 * Bandwidth  # 采样率 [Hz]
h = 800e3  # 卫星高度 [m]
G0 = 10 ** (40 / 10)  # 天线增益
theta_w = np.deg2rad(1.2)  # 天线波束宽度 [rad]
R_earth = 6371e3  # 地球半径 [m]

# 海况参数（真实值）
SWH_true = 2.0  # 有效波高 [m]
xi_true = np.deg2rad(0.2)  # 离天底角 / 误指向角 [rad]
lambda_s = -0.1  # 海面偏度 (用于生成)
kappa_s = 0.1  # 海面峰度 (用于生成)

# 时间轴设置
t_span = 300e-9  # 总时间跨度 [s]
Nt = 4096  # 采样点数
t_center = 0  # 以星下点回波为参考零点
t_start = t_center - t_span / 2
t_end = t_center + t_span / 2
t = np.linspace(t_start, t_end, Nt)  # 时间轴 [s]
dt = t[1] - t[0]


# ========== 三项卷积模型类 ==========
class ThreeConvolutionModel:
    def __init__(self, c, h, theta_w, Bandwidth, lambda_s, kappa_s, t, dt):
        self.c = c
        self.h = h
        self.theta_w = theta_w
        self.Bandwidth = Bandwidth
        self.lambda_s = lambda_s
        self.kappa_s = kappa_s
        self.t = t
        self.dt = dt
        self.t_delay_nadir = 2 * h / c
        self.gamma = 2 * np.log(4) / (np.sin(theta_w / 2) ** 2)

    def hermite_poly(self, n, z):
        if n == 3:
            return z ** 3 - 3 * z
        elif n == 4:
            return z ** 4 - 6 * z ** 2 + 3
        elif n == 6:
            return z ** 6 - 15 * z ** 4 + 45 * z ** 2 - 15
        else:
            raise ValueError("只实现 H3, H4, H6")

    def generate_waveform(self, tau, P_u, SWH, xi):
        """
        生成三项卷积回波波形

        参数:
            tau: 回波延时 [s] (相对于星下点的延时)
            P_u: 回波幅度
            SWH: 有效波高 [m]
            xi: 离天底角 / 误指向角 [rad]

        返回:
            waveform: 归一化后的回波波形
        """
        # 指向点回波时延（绝对时间）
        pointing_delay = tau

        # 平坦海面冲激响应
        P_FS = np.zeros_like(self.t)
        r_t = self.c * (self.t + self.t_delay_nadir) / 2
        valid_mask = r_t >= self.h
        rho_t = np.zeros_like(self.t)
        rho_t[valid_mask] = np.sqrt(r_t[valid_mask] ** 2 - self.h ** 2)

        theta_t = np.zeros_like(self.t)
        theta_t[valid_mask] = np.arctan(rho_t[valid_mask] / self.h) - xi

        P_FS_temp = np.zeros_like(self.t)
        if np.any(valid_mask):
            G_sq = np.exp(-(4 / self.gamma) * np.sin(theta_t[valid_mask]) ** 2)
            range_factor = 1 / r_t[valid_mask] ** 4
            jacobian = np.zeros_like(rho_t[valid_mask])
            nonzero_rho_mask = rho_t[valid_mask] > 1e-6
            jacobian[nonzero_rho_mask] = (self.c / 2) * (
                        r_t[valid_mask][nonzero_rho_mask] / rho_t[valid_mask][nonzero_rho_mask])
            P_FS_temp[valid_mask] = G_sq * range_factor * jacobian

        if np.max(P_FS_temp) > 0:
            P_FS = P_FS_temp / np.max(P_FS_temp)

        # 海面高程分布
        sigma_s = SWH / 4 * (2 / self.c)
        z_t = self.t / sigma_s
        gaussian = np.exp(-0.5 * z_t ** 2) / (np.sqrt(2 * np.pi) * sigma_s)
        H3 = self.hermite_poly(3, z_t)
        H4 = self.hermite_poly(4, z_t)
        H6 = self.hermite_poly(6, z_t)
        q_s = gaussian * (1 + (self.lambda_s / 6) * H3 + (self.kappa_s / 24) * H4 + (self.lambda_s ** 2 / 72) * H6)
        q_s_integral = np.trapz(q_s, self.t)
        if q_s_integral > 0:
            q_s = q_s / q_s_integral

        # 系统响应
        sigma_r = 1 / (2 * self.Bandwidth)
        s_r = np.exp(-0.5 * ((self.t - pointing_delay) / sigma_r) ** 2)
        s_r_integral = np.trapz(s_r, self.t)
        if s_r_integral > 0:
            s_r = s_r / s_r_integral

        # 三项卷积
        P_FS = np.nan_to_num(P_FS)
        q_s = np.nan_to_num(q_s)
        s_r = np.nan_to_num(s_r)

        W1 = convolve(P_FS, q_s, mode='same') * self.dt
        W_final = convolve(W1, s_r, mode='same') * self.dt
        if np.max(W_final) > 0:
            W_final = W_final / np.max(W_final)

        return P_u * W_final


# ========== 1. 生成观测回波（使用三项卷积模型） ==========
print("生成三项卷积回波模型...")

# 创建三项卷积模型实例
true_model = ThreeConvolutionModel(c, h, theta_w, Bandwidth, lambda_s, kappa_s, t, dt)

# 计算真实延时
t_delay_nadir = 2 * h / c
rho_min = h * np.tan(xi_true)
r_min = np.sqrt(h ** 2 + rho_min ** 2)
t_delay_min = 2 * r_min / c
pointing_delay = t_delay_min - t_delay_nadir
tau_true = pointing_delay
P_u_true = 1.0

# 生成观测波形
observed_waveform = true_model.generate_waveform(tau_true, P_u_true, SWH_true, xi_true)

print(f"真实参数: tau={tau_true * 1e9:.2f} ns, P_u={P_u_true:.3f}, SWH={SWH_true:.2f} m, xi={np.rad2deg(xi_true):.2f}°")


# ========== 2. 改进的加权最小二乘算法（使用三项卷积模型） ==========
class RadarAltimeterRetracker:
    def __init__(self, forward_model):
        self.forward_model = forward_model
        self.t = forward_model.t
        self.Nt = len(self.t)
        self.t_start = self.t[0]
        self.t_end = self.t[-1]
        self.c = forward_model.c

    def estimate_initial_params_from_waveform(self, t, waveform):
        peak_idx = np.argmax(waveform)
        half_power = 0.5 * waveform[peak_idx]
        front_indices = np.where(waveform[:peak_idx] > half_power)[0]
        if len(front_indices) > 0:
            tau_init = t[front_indices[0]]
        else:
            tau_init = t[peak_idx] - 10e-9
        P_u_init = np.max(waveform)
        if peak_idx > 5:
            p10 = 0.1 * P_u_init
            p90 = 0.9 * P_u_init
            idx10 = np.where(waveform[:peak_idx] > p10)[0]
            idx90 = np.where(waveform[:peak_idx] > p90)[0]
            if len(idx10) > 0 and len(idx90) > 0:
                rise_time = t[idx90[0]] - t[idx10[0]]
                SWH_init = rise_time * self.c / 2 * 4
                SWH_init = np.clip(SWH_init, 0.5, 10.0)
            else:
                SWH_init = 2.0
        else:
            SWH_init = 2.0
        xi_init = 0.0
        return [tau_init, P_u_init, SWH_init, xi_init]

    def weighted_least_squares_retracking(self, observed_waveform, initial_params,
                                          max_iterations=200, tolerance=1e-10,
                                          learning_rate_init=1e-10, use_line_search=True):
        tau0, P_u0, SWH0, xi0 = initial_params
        N_noise0 = 0
        current_params = np.array([tau0, P_u0, SWH0, xi0, N_noise0])
        history = {'params': [current_params.copy()], 'cost': [], 'iterations': 0}
        learning_rate = learning_rate_init

        for iteration in range(max_iterations):
            tau, P_u, SWH, xi, N_noise = current_params
            model_waveform = self.forward_model.generate_waveform(tau, P_u, SWH, xi)
            residuals = observed_waveform - model_waveform

            signal_power = np.maximum(0.01 * P_u, np.abs(model_waveform))
            residual_power = np.abs(residuals)
            weights = 1.0 / (signal_power + 0.1 * residual_power)
            weights = weights / np.sum(weights) * len(weights)
            P_matrix = np.diag(weights)

            current_cost = np.sum(weights * residuals ** 2)
            history['cost'].append(current_cost)

            if iteration > 0:
                rel_change = abs(history['cost'][-2] - current_cost) / max(history['cost'][-2], 1e-12)
                if rel_change < tolerance:
                    print(f"收敛于迭代 {iteration}, 相对变化 {rel_change:.2e}, 代价 {current_cost:.6e}")
                    break

            scales = np.array([1e9, 1.0, 1.0, 1e3])
            jacobian = np.zeros((self.Nt, 4))
            base = model_waveform

            delta_val = 1e-6
            delta_tau = delta_val / scales[0]
            base_tau = self.forward_model.generate_waveform(tau + delta_tau, P_u, SWH, xi)
            jacobian[:, 0] = (base_tau - base) / delta_tau * scales[0]

            base_Pu = self.forward_model.generate_waveform(tau, P_u + delta_val, SWH, xi)
            jacobian[:, 1] = (base_Pu - base) / delta_val

            base_SWH = self.forward_model.generate_waveform(tau, P_u, SWH + delta_val, xi)
            jacobian[:, 2] = (base_SWH - base) / delta_val

            delta_xi = delta_val / scales[3]
            base_xi = self.forward_model.generate_waveform(tau, P_u, SWH, xi + delta_xi)
            jacobian[:, 3] = (base_xi - base) / delta_xi * scales[3]

            B = jacobian
            P = P_matrix
            V = residuals

            try:
                left_matrix = B.T @ P @ B
                right_vector = B.T @ P @ V
                reg = 1e-8 * np.eye(left_matrix.shape[0])
                delta_full = np.linalg.solve(left_matrix + reg, right_vector)
                if np.abs(delta_full[2]) > 5.0:
                    delta_full[2] = np.sign(delta_full[2]) * 5.0
            except np.linalg.LinAlgError:
                gradient = -2 * B.T @ P @ V
                delta_full = -learning_rate * gradient / (np.linalg.norm(gradient) + 1e-12)

            if use_line_search:
                alpha = 1.0
                best_alpha = alpha
                best_cost = current_cost
                best_delta = delta_full.copy()

                for _ in range(10):
                    test_params = current_params.copy()
                    test_params[:4] += alpha * delta_full[:4]
                    test_waveform = self.forward_model.generate_waveform(
                        test_params[0], test_params[1], test_params[2], test_params[3])
                    test_residuals = observed_waveform - test_waveform
                    test_cost = np.sum(weights * test_residuals ** 2)

                    if test_cost < best_cost:
                        best_cost = test_cost
                        best_alpha = alpha
                        best_delta = alpha * delta_full
                        break
                    alpha *= 0.5
                delta_full[:4] = best_delta[:4]
            else:
                if iteration > 0:
                    if current_cost > history['cost'][-2]:
                        learning_rate *= 0.7
                    else:
                        learning_rate *= 1.2
                learning_rate = np.clip(learning_rate, 1e-10, 1e-6)

                grad_norm = np.linalg.norm(delta_full[:4])
                if grad_norm > 1e-12:
                    delta_full[:4] = learning_rate * delta_full[:4] / grad_norm

            current_params[:4] += delta_full[:4]

            current_params[0] = np.clip(current_params[0], self.t_start + 1e-9, self.t_end - 1e-9)
            current_params[1] = np.clip(current_params[1], 0.01, 2.0)
            current_params[2] = np.clip(current_params[2], 0.01, 20.0)
            current_params[3] = np.clip(current_params[3], -0.05, 0.05)

            history['params'].append(current_params.copy())
            history['iterations'] = iteration + 1

            if iteration % 20 == 0:
                print(f"迭代 {iteration}: cost={current_cost:.3e}, "
                      f"tau={current_params[0] * 1e9:.2f}ns, "
                      f"P_u={current_params[1]:.3f}, "
                      f"SWH={current_params[2]:.2f}m, "
                      f"xi={np.rad2deg(current_params[3]):.2f}°, "
                      f"lr={learning_rate:.2e}")

        return current_params[:4], history


# 创建重跟踪器实例（使用相同的三项卷积模型）
retracker = RadarAltimeterRetracker(true_model)

# 从波形估计初始参数
initial_params = retracker.estimate_initial_params_from_waveform(t, observed_waveform)
print(f"\n初始参数（初猜）: tau={initial_params[0] * 1e9:.2f}ns, "
      f"P_u={initial_params[1]:.3f}, SWH={initial_params[2]:.2f}m, "
      f"xi={np.rad2deg(initial_params[3]):.2f}°")

# 运行改进的加权最小二乘
print("\n开始加权最小二乘反演...")
start = time.time()
estimated_params, history = retracker.weighted_least_squares_retracking(
    observed_waveform,
    initial_params,
    max_iterations=300,
    tolerance=1e-10,
    learning_rate_init=1e-3,
    use_line_search=True
)
elapsed = time.time() - start

tau_est, P_u_est, SWH_est, xi_est = estimated_params
print(f"\n反演完成，耗时 {elapsed:.2f} 秒")
print(f"估计参数: tau={tau_est * 1e9:.2f}ns, P_u={P_u_est:.3f}, SWH={SWH_est:.2f}m, xi={np.rad2deg(xi_est):.2f}°")

# ========== 3. 绘制三张图 ==========

t_ns = t * 1e9

# 图1：三项卷积生成的观测回波
plt.figure(figsize=(8, 5))
plt.plot(t_ns, observed_waveform, 'b-', linewidth=2, label='观测回波 (三项卷积)')
plt.axvline(x=tau_true * 1e9, color='k', linestyle='--', label=f'真实延时 τ = {tau_true * 1e9:.1f} ns')
plt.xlabel('时间 [ns] (星下点回波 = 0)')
plt.ylabel('归一化功率')
plt.title('三项卷积模型生成的观测回波')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('observed_waveform.png', dpi=150)
plt.show()

# 图2：迭代收敛过程
plt.figure(figsize=(8, 5))
plt.plot(history['cost'], 'b-', linewidth=2)
plt.xlabel('迭代次数')
plt.ylabel('代价函数值')
plt.title('加权最小二乘迭代收敛过程')
plt.yscale('log')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('convergence.png', dpi=150)
plt.show()

# 图3：最终估计参数与真值的对比误差
params_name = ['时延 τ (ns)', '幅度 P_u', '波高 SWH (m)', '指向角 ξ (°)']
true_vals = [tau_true * 1e9, P_u_true, SWH_true, np.rad2deg(xi_true)]
est_vals = [tau_est * 1e9, P_u_est, SWH_est, np.rad2deg(xi_est)]
errors = [est - true for est, true in zip(est_vals, true_vals)]
relative_errors = [err / true if true != 0 else 0 for err, true in zip(errors, true_vals)]

x = np.arange(len(params_name))
width = 0.35

plt.figure(figsize=(10, 6))
bars1 = plt.bar(x - width / 2, true_vals, width, label='真实值', alpha=0.7)
bars2 = plt.bar(x + width / 2, est_vals, width, label='估计值', alpha=0.7)
plt.xlabel('参数')
plt.ylabel('数值')
plt.title('参数估计值与真实值对比')
plt.xticks(x, params_name)
plt.legend()
for i, (true, est) in enumerate(zip(true_vals, est_vals)):
    plt.text(i - width / 2, true + 0.02 * max(true_vals), f'{true:.2f}', ha='center', va='bottom', fontsize=9)
    plt.text(i + width / 2, est + 0.02 * max(true_vals), f'{est:.2f}', ha='center', va='bottom', fontsize=9)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('estimation_error.png', dpi=150)
plt.show()

# 打印详细误差
print("\n参数估计误差:")
print("-" * 50)
print(f"{'参数':<15} {'真实值':<10} {'估计值':<10} {'绝对误差':<12} {'相对误差':<10}")
for name, true, est, err, rel in zip(params_name, true_vals, est_vals, errors, relative_errors):
    print(f"{name:<15} {true:<10.3f} {est:<10.3f} {err:<+12.3f} {rel * 100:>+9.1f}%")
print("-" * 50)

# ========== 4. 最终拟合波形与真值波形对比 ==========
print("\n生成最终拟合波形对比图...")

# 使用估计参数生成拟合波形（使用相同的三项卷积模型）
fitted_waveform = true_model.generate_waveform(tau_est, P_u_est, SWH_est, xi_est)

plt.figure(figsize=(10, 6))
plt.plot(t_ns, observed_waveform, 'b-', linewidth=2, label='真实波形 (三项卷积)')
plt.plot(t_ns, fitted_waveform, 'r--', linewidth=2, label='拟合波形 (三项卷积)')
plt.axvline(x=tau_true * 1e9, color='b', linestyle=':', alpha=0.5, label=f'真实τ={tau_true * 1e9:.2f}ns')
plt.axvline(x=tau_est * 1e9, color='r', linestyle=':', alpha=0.5, label=f'估计τ={tau_est * 1e9:.2f}ns')
plt.xlabel('时间 [ns] (星下点回波 = 0)')
plt.ylabel('归一化功率')
plt.title('最终拟合波形与真实波形对比')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('waveform_comparison.png', dpi=150)
plt.show()

# 打印拟合精度
print("\n波形拟合精度:")
print("-" * 50)
rmse = np.sqrt(np.mean((observed_waveform - fitted_waveform) ** 2))
print(f"均方根误差 (RMSE): {rmse:.6f}")
ss_res = np.sum((observed_waveform - fitted_waveform) ** 2)
ss_tot = np.sum((observed_waveform - np.mean(observed_waveform)) ** 2)
r2 = 1 - (ss_res / ss_tot)
print(f"决定系数 R²: {r2:.6f}")
print("-" * 50)