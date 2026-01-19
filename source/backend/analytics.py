import pandas as pd


# =========================================================
# Load data
# =========================================================
def load_data():
    output_s3_path = "s3://startel/startel_output_csv/part-*.csv"

    df = pd.read_csv(output_s3_path ,
    storage_options={"expand": True})

# Split month name and year
    df["billing_month_name"] = df["billing_month"].str.extract(r"([A-Za-z]+)")
    df["billing_year"] = df["billing_month"].str.extract(r"(\d{4})").astype(int)


# df = df.drop(columns=["billing_month"])
    df = df.drop(columns=["billing_month"], errors="ignore")
    df = df.rename(columns={"billing_month_name": "billing_month"})
# Month ordering for correct time analysis
    month_order = {
        "January": 1, "February": 2, "March": 3, "April": 4,
        "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12
}

    df["month_num"] = df["billing_month"].map(month_order)
# Sort properly
    df = df.sort_values(
    ["customer_id", "billing_year", "month_num"]
    ).reset_index(drop=True)

    plan_rank = {"Silver": 1, "Gold": 2, "Platinum": 3}
    df["plan_rank"] = df["plan"].map(plan_rank)

# Previous state
    df["prev_plan"] = df.groupby("customer_id")["plan"].shift(1)
    df["prev_plan_rank"] = df.groupby("customer_id")["plan_rank"].shift(1)
# Change direction
    df["plan_change"] = df["plan_rank"] - df["prev_plan_rank"]

    def plan_movement(x):
        if pd.isna(x):
         return "new_customer"
        elif x > 0:
            return "upgrade"
        elif x < 0:
            return "downgrade"
        else:
            return "no_change"

    df["movement_type"] = df["plan_change"].apply(plan_movement)

    df["transition"] = df["prev_plan"].fillna("NONE") + "_to_" + df["plan"]

# -------------------------------------------------
# STEP 4: ROBUST CHURN LOGIC (missing-invoice safe)
# -------------------------------------------------

# Create a continuous time index
    df["time_index"] = df["billing_year"] * 12 + df["month_num"]

# Identify last activity per customer
    last_activity = df.groupby("customer_id").agg(
        last_seen_time=("time_index", "max")
).reset_index()

# Dataset end (observation window end)
    global_last_time = df["time_index"].max()
# Months since last invoice
    last_activity["months_since_last_seen"] = (
    global_last_time - last_activity["last_seen_time"]
)

# Churn threshold (industry-safe)
    CHURN_GAP = 6   # months

# Final churn label
    last_activity["is_churned"] = (
    last_activity["months_since_last_seen"] >= CHURN_GAP
)

# Optional: churn confidence (0–1)
    last_activity["churn_confidence"] = (
    last_activity["months_since_last_seen"] / CHURN_GAP
    ).clip(0, 1)

# Attach churn info back to main dataframe
    df = df.merge(last_activity, on="customer_id", how="left")

#step 5 - CUSTOMER REVENUE FEATURES
    customer_revenue = df.groupby("customer_id").agg(
        total_paid=("bill_due", "sum"),
    avg_monthly_bill=("bill_due", "mean"),
    active_months=("bill_due", "count"),
    max_bill=("bill_due", "max"),
    min_bill=("bill_due", "min")
    ).reset_index()

# STEP 6 — CITY & PLAN INTELLIGENCE
    city_summary = df.groupby("city").agg(
    total_users=("customer_id", "nunique"),
    total_revenue=("bill_due", "sum"),
    avg_bill=("bill_due", "mean")
    ).reset_index()


    events_df = df
    customer_df = customer_revenue
    city_df = city_summary


    return events_df, city_df, customer_df


# =========================================================
# Helpers
# =========================================================
def normalize(text):
    return (
        str(text).lower()
        .replace("→", " ")
        .replace("_", " ")
        .replace("-", " ")
        .replace("to", " ")
        .replace("  ", " ")
        .strip()
    )


def get_amount_column(df):
    for c in df.columns:
        name = c.lower()
        if "amount" in name or "bill" in name or "revenue" in name or "paid" in name:
            return c
    raise ValueError("Revenue column not found")


# =========================================================
# MAIN ANALYTICS ENGINE
# =========================================================
def answer_analytical(question, events_df, city_df, customer_df):

    q = question.lower()
    q_norm = normalize(q)

    # detect year
    years = [int(w) for w in q.split() if w.isdigit()]
    year = years[0] if years else None

    amount_col = get_amount_column(events_df)

    df = events_df.copy()
    if year:
        df = df[df["billing_year"] == year]

    # =====================================================
    # UPGRADE / DOWNGRADE (REAL, ROBUST)
    # =====================================================
    if "upgrade" in q or "downgrade" in q:

        # direction
        if "upgrade" in q:
            df = df[df["plan_change"] == 1]
            action = "upgraded"
        else:
            df = df[df["plan_change"] == -1]
            action = "downgraded"

        # detect plans from question
        plans = ["silver", "gold", "platinum"]
        from_plan = None
        to_plan = None

        for p in plans:
            if p in q_norm:
                if from_plan is None:
                    from_plan = p
                else:
                    to_plan = p

        # if both plans detected → transition-level answer
        if from_plan and to_plan:
            mask = (
                df["transition"].str.lower().str.contains(from_plan)
                & df["transition"].str.lower().str.contains(to_plan)
            )
            count = df[mask].shape[0]

            suffix = f" in {year}" if year else ""
            return f"{count} customers {action} from {from_plan} to {to_plan}{suffix}."

        # else → total upgrades / downgrades
        suffix = f" in {year}" if year else ""
        return f"{df.shape[0]} customers {action}{suffix}."

    # =====================================================
    # LIST ALL CITIES
    # =====================================================
    if "city" in q and ("list" in q or "what all" in q or "all" in q):
        cities = sorted(city_df["city"].dropna().unique().tolist())
        return f"The cities are: {', '.join(cities)}"

    # =====================================================
    # TOTAL REVENUE (YEAR)
    # =====================================================
    if "total revenue" in q and year:
        total = df[amount_col].sum()
        return f"Total revenue in {year} is {total:.2f}"

    # =====================================================
    # HIGHEST REVENUE YEAR
    # =====================================================
    if "highest revenue" in q or "max revenue" in q:
        yearly = events_df.groupby("billing_year")[amount_col].sum()
        y = yearly.idxmax()
        return f"The highest revenue was in {y} with {yearly[y]:.2f}"

    # =====================================================
    # TOP CUSTOMER
    # =====================================================
    if "top customer" in q or "highest contributor" in q:
        top = customer_df.sort_values("total_paid", ascending=False).iloc[0]
        return (
            f"Customer ID {top['customer_id']} is the highest contributor "
            f"with total payment of {top['total_paid']:.2f}"
        )

    return "This analytical question is not supported yet."
