import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

st.title("🔍 Bayesian Coin Bias Estimator Using Bayes’ Theorem")
st.write("""
This app uses **Bayes' Theorem** to estimate the bias (probability of Heads) 
of a coin after observing some flips.
""")

# ---- INPUTS ----
st.sidebar.header("Input Parameters")

n = st.sidebar.number_input("Number of flips (n)", min_value=1, value=20)
k = st.sidebar.number_input("Number of heads (k)", min_value=0, max_value=n, value=12)

prior_choice = st.sidebar.selectbox(
    "Choose Prior Distribution",
    ["Uniform (Beta 1,1)", "Beta(2,2)", "Beta(5,5)"]
)

# Prior parameters
if prior_choice == "Uniform (Beta 1,1)":
    a0, b0 = 1, 1
elif prior_choice == "Beta(2,2)":
    a0, b0 = 2, 2
else:
    a0, b0 = 5, 5

# ---- POSTERIOR ----
posterior_a = a0 + k
posterior_b = b0 + (n - k)

# ---- PLOT ----
theta = np.linspace(0, 1, 500)
prior_pdf = beta.pdf(theta, a0, b0)
posterior_pdf = beta.pdf(theta, posterior_a, posterior_b)

fig, ax = plt.subplots()
ax.plot(theta, prior_pdf, label="Prior", linestyle="--")
ax.plot(theta, posterior_pdf, label="Posterior", linewidth=2)
ax.set_title("Prior vs Posterior Distribution")
ax.set_xlabel("θ (Probability of Heads)")
ax.set_ylabel("Density")
ax.legend()

st.pyplot(fig)

# ---- RESULTS ----
st.subheader("📊 Posterior Estimates")

posterior_mean = posterior_a / (posterior_a + posterior_b)
posterior_map = (posterior_a - 1) / (posterior_a + posterior_b - 2)

st.write(f"**Posterior Mean Estimate:** {posterior_mean:.4f}")
st.write(f"**MAP Estimate:** {posterior_map:.4f}")

st.info("""
### How it works
- Prior belief: θ ~ Beta(a₀, b₀)
- Likelihood: k heads out of n flips
- Posterior: θ | data ~ Beta(a₀ + k, b₀ + n − k)

This is a direct application of **Bayes' Theorem** using Beta–Binomial conjugacy.
""")
