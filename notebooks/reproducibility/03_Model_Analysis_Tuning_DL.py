# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # Deep Learning Model Analysis and Fine-Tuning

# %% [markdown]
# ## 1. Introduction and Baseline Review
# In the previous notebook, we trained a variety of deep learning baselines and performed a Phase 2a grid search on Transfer Learning initialization strategies for our champion, MobileNetV2.
#
# ## 2. Preliminary Error Analysis (Phase 2a Champion)
# Before proceeding with optimizer fine-tuning (Phase 2b), we must analyze the errors of our current best model (`mobilenet_v2_level_1_imagenet_dr0.0`). Since we rigorously logged our test metrics to WandB, we can leverage the W&B Jupyter integrations directly here to visualize our confusion matrices automatically.
#
# This allows us to verify where our model is systematically failing, and check whether it's struggling with certain class overlaps (e.g., Macrophomina vs. Stem Rot) or fundamental dataset labeling quality, before we invest further compute in finding the perfect gradient descent parameters.

# %%
import IPython.display

IPython.display.IFrame(
    src="https://wandb.ai/grade-descent/jute-disease-detection",
    width="100%",
    height="600px",
)

# %% [markdown]
# ## 3. Phase 2b: Optimizer Fine-Tuning Grid Search (MobileNetV2)
# With our baseline errors confirmed as expected architectural characteristics (and not systemic labeling bugs), we can comfortably execute Phase 2b. We will sweep across standard Learning Rates and Weight Decays to squeeze the ultimate convergence trajectory out of our Jute dataset.

# %%
# !make grid-search-finetune
