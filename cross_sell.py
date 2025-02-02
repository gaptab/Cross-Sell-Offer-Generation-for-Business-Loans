import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
# ✅ Step 1: Generate Dummy Data
np.random.seed(42)

num_customers = 500  # Number of customers
customer_data = pd.DataFrame({
    "Customer_ID": [f"CUST_{i}" for i in range(1, num_customers + 1)],
    "Credit_Score": np.random.randint(500, 900, num_customers),  # Credit Score (500-900)
    "Annual_Income": np.random.randint(20000, 200000, num_customers),  # Income (20K - 200K)
    "Existing_Loan_Amount": np.random.randint(0, 50000, num_customers),  # Existing Loan Balance
    "Loan_Defaults": np.random.randint(0, 3, num_customers),  # Past Loan Defaults (0-2)
    "Business_Revenue": np.random.randint(50000, 1000000, num_customers),  # Business Revenue
    "Industry_Type": np.random.choice(["Retail", "Manufacturing", "Tech", "Services"], num_customers),
    "Years_in_Business": np.random.randint(1, 30, num_customers),  # Business Age (1-30 years)
})

# ✅ Step 2: Feature Engineering & Cleaning
customer_data["Debt_to_Income_Ratio"] = customer_data["Existing_Loan_Amount"] / customer_data["Annual_Income"]
customer_data["Debt_to_Income_Ratio"].replace([np.inf, -np.inf], np.nan, inplace=True)  # Fix Inf values
customer_data.fillna(0, inplace=True)  # Replace NaN with 0

# ✅ Step 3: Loan Offer Calculation Logic
def calculate_loan_offer(row):
    """Logic to determine how much business loan to offer a customer"""
    if row["Credit_Score"] > 750 and row["Loan_Defaults"] == 0:
        return min(row["Annual_Income"] * 0.4, 50000)  # High Credit Score → Offer 40% of Income (max 50K)
    elif row["Credit_Score"] > 650 and row["Loan_Defaults"] <= 1:
        return min(row["Annual_Income"] * 0.25, 30000)  # Medium Credit Score → Offer 25% of Income (max 30K)
    else:
        return 0  # Low Credit Score or Multiple Defaults → No Offer

customer_data["Loan_Offer_Amount"] = customer_data.apply(calculate_loan_offer, axis=1)

# ✅ Step 4: Customer Segmentation
def categorize_customer(row):
    """Segment customers based on eligibility"""
    if row["Loan_Offer_Amount"] >= 40000:
        return "Premium Offer"
    elif row["Loan_Offer_Amount"] >= 20000:
        return "Standard Offer"
    elif row["Loan_Offer_Amount"] > 0:
        return "Basic Offer"
    else:
        return "Not Eligible"

customer_data["Segment"] = customer_data.apply(categorize_customer, axis=1)

# ✅ Step 5: Save Data to CSV
customer_data.to_csv("Cross_Sell_Loan_Offers.csv", index=False)
print("✅ Cross-sell loan offers data saved successfully!")

# ✅ Step 6: Summary Report
summary = customer_data["Segment"].value_counts()
print("\n📊 Customer Segmentation Summary:\n", summary)



# Load data
customer_data = pd.read_csv("Cross_Sell_Loan_Offers.csv")

# 🎯 1️⃣ Customer Segmentation Distribution (Bar Chart)
plt.figure(figsize=(8, 5))
sns.countplot(data=customer_data, x="Segment", palette="viridis", order=["Premium Offer", "Standard Offer", "Basic Offer", "Not Eligible"])
plt.title("Customer Segmentation Distribution", fontsize=14)
plt.xlabel("Loan Offer Segment")
plt.ylabel("Number of Customers")
plt.xticks(rotation=30)
plt.show()

# 🎯 2️⃣ Loan Offer Amount vs. Credit Score (Scatter Plot)
plt.figure(figsize=(8, 5))
sns.scatterplot(data=customer_data, x="Credit_Score", y="Loan_Offer_Amount", hue="Segment", palette="coolwarm", alpha=0.7)
plt.title("Loan Offer Amount vs. Credit Score", fontsize=14)
plt.xlabel("Credit Score")
plt.ylabel("Loan Offer Amount ($)")
plt.axvline(x=750, color="green", linestyle="--", label="High Credit Score Threshold")
plt.legend()
plt.show()

# 🎯 3️⃣ Debt-to-Income Ratio Distribution (Histogram)
plt.figure(figsize=(8, 5))
sns.histplot(customer_data["Debt_to_Income_Ratio"], bins=30, kde=True, color="blue")
plt.title("Debt-to-Income Ratio Distribution", fontsize=14)
plt.xlabel("Debt-to-Income Ratio")
plt.ylabel("Number of Customers")
plt.show()

# 🎯 4️⃣ Loan Offer Amount by Industry Type (Box Plot)
plt.figure(figsize=(8, 5))
sns.boxplot(data=customer_data, x="Industry_Type", y="Loan_Offer_Amount", palette="Set2")
plt.title("Loan Offer Amount by Industry Type", fontsize=14)
plt.xlabel("Industry Type")
plt.ylabel("Loan Offer Amount ($)")
plt.xticks(rotation=30)
plt.show()

