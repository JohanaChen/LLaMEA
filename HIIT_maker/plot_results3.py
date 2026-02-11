import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---- Load data ----
path = "Results/Experiment_3/HIIT_2/Results2.xlsx"
df = pd.read_excel(path)

# ---- Column names ----
interval_col = "Number of exercise"
pred_col = "Average HR"

user_cols = [c for c in df.columns if str(c).lower().startswith("user")]

# ---- Compute observed mean HR across users ----
df["observed_mean_hr"] = df[user_cols].mean(axis=1, skipna=True)

# ---- Compute error per interval (Predicted - Observed mean) ----
df["error"] = df[pred_col] - df["observed_mean_hr"]

# ---- Aggregate error metrics ----
mean_error = df["error"].mean(skipna=True)
mae = df["error"].abs().mean(skipna=True)
rmse = np.sqrt((df["error"] ** 2).mean(skipna=True))

print(f"Mean Error (ME): {mean_error:.2f} BPM")
print(f"Mean Absolute Error (MAE): {mae:.2f} BPM")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f} BPM")

# ---- X axis: intervals in order ----
x = np.arange(len(df))
x_labels = df[interval_col].astype(str)

# ---- Plot ----
plt.figure(figsize=(12, 6))

# Predicted average HR (bold red)
plt.plot(
    x,
    df[pred_col],
    color="red",
    linewidth=3,
    label="Predicted HR (average)"
)

# Collected HR from users (lighter lines)
for user in user_cols:
    plt.plot(
        x,
        df[user],
        linewidth=1.5,
        alpha=0.4,
        label=user
    )

# ---- Formatting ----
plt.xticks(x, x_labels, rotation=45, ha="right")
plt.xlabel("HIIT interval timeline")
plt.ylabel("Heart rate (BPM)")
plt.grid(True, alpha=0.3)

# Avoid legend clutter: keep predicted + users grouped
plt.legend(ncol=2, fontsize=9)
plt.tight_layout()

# ---- Show / save ----
plt.show()
# plt.savefig("experiment3_predicted_vs_collected_hr.png", dpi=300, bbox_inches="tight")


