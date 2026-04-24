import matplotlib.pyplot as plt
import numpy as np

# --- Load + clean data ---
with open("rewards.txt", "r") as f:
    rewards = [float(l.strip().replace("'", "")) for l in f]

# --- Exponential Moving Average ---
def ema(data, alpha=0.1):
    result = []
    ema_val = data[0]
    for x in data:
        ema_val = alpha * x + (1 - alpha) * ema_val
        result.append(ema_val)
    return result

# smooth curves
ema_fast = ema(rewards, alpha=0.2)   # less smooth
ema_slow = ema(rewards, alpha=0.1)   # more smooth
ema_slow2 = ema(rewards, alpha=0.05)
# x-axis
episodes = np.arange(1, len(rewards) + 1)

# --- Plot ---
plt.figure(figsize=(10, 5))

# raw rewards (light + thin)
plt.plot(episodes, rewards, alpha=0.15, linewidth=1, label="Actual Rewards")

# smoothed curves
plt.plot(episodes, ema_fast, linewidth=2, label="EMA (fast)")
plt.plot(episodes, ema_slow, linewidth=3, label="EMA (slow)")
plt.plot(episodes, ema_slow2, linewidth=4, label="EMA (slow)")


# labels + style
plt.title("Reward Curve", fontsize=14)
plt.xlabel("Episode")
plt.ylabel("Reward")

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()