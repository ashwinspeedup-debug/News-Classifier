"""
save_artifacts.py
─────────────────
Run this script AFTER training your model in the notebook to export
the model and tokenizer so the deployment apps can load them.

Usage:
    python save_artifacts.py
"""

import pickle

# ── If running inside the notebook, paste the cells below directly ──────────

# 1.  Save the Keras model
#     Replace `model` with your final trained model variable name.
model.save("model.h5")
print("✅ model.h5 saved.")

# 2.  Save the Keras tokenizer
#     Replace `tokenizer` with your tokenizer variable name.
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
print("✅ tokenizer.pkl saved.")

# 3.  (Optional) Save the LabelEncoder if needed
# with open("label_encoder.pkl", "wb") as f:
#     pickle.dump(encoder, f)
# print("✅ label_encoder.pkl saved.")

print("\nDone! Copy model.h5 and tokenizer.pkl into your deployment folder.")
