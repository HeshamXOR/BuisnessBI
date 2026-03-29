"""
Dataset Generation Script
==========================
Generates 4 realistic synthetic CSV datasets for business and tech analysis.
Run this script to create the datasets in the data/ directory.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_sales_data(n_rows: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Generate realistic sales transaction data."""
    np.random.seed(seed)

    products = {
        "Electronics": ["Laptop Pro X", "Wireless Headphones", "Smart Watch Ultra",
                        "4K Monitor", "Mechanical Keyboard", "USB-C Hub"],
        "Software": ["Cloud Suite License", "Security Package", "Analytics Platform",
                      "DevOps Toolkit", "AI Copilot Pro"],
        "Services": ["Consulting Package", "Training Workshop", "Support Plan Premium",
                      "Migration Service", "Audit & Compliance"],
        "Hardware": ["Server Rack Unit", "Network Switch", "SSD Storage 2TB",
                     "GPU Accelerator Card", "Smart IoT Gateway"]
    }

    regions = ["North America", "Europe", "Asia Pacific", "Middle East", "Latin America"]
    region_weights = [0.35, 0.28, 0.22, 0.08, 0.07]

    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=np.random.randint(0, 730)) for _ in range(n_rows)]

    records = []
    for i in range(n_rows):
        category = np.random.choice(list(products.keys()))
        product = np.random.choice(products[category])
        region = np.random.choice(regions, p=region_weights)

        base_revenue = {
            "Electronics": np.random.uniform(150, 2500),
            "Software": np.random.uniform(200, 5000),
            "Services": np.random.uniform(500, 15000),
            "Hardware": np.random.uniform(300, 8000)
        }[category]

        # Add seasonal variation
        month = dates[i].month
        seasonal_factor = 1.0 + 0.15 * np.sin(2 * np.pi * (month - 3) / 12)
        revenue = round(base_revenue * seasonal_factor, 2)

        units = max(1, int(np.random.exponential(5)))
        profit_margin = round(np.random.uniform(0.08, 0.45), 2)
        cost = round(revenue * (1 - profit_margin), 2)
        discount = round(np.random.choice([0, 0, 0, 0.05, 0.10, 0.15, 0.20]), 2)

        records.append({
            "date": dates[i].strftime("%Y-%m-%d"),
            "product": product,
            "category": category,
            "region": region,
            "revenue": revenue,
            "units_sold": units,
            "cost": cost,
            "profit_margin": profit_margin,
            "discount_applied": discount
        })

    df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    return df


def generate_marketing_data(n_rows: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generate realistic marketing campaign data."""
    np.random.seed(seed + 1)

    campaign_types = ["Brand Awareness", "Lead Generation", "Product Launch",
                      "Retargeting", "Seasonal Promo", "Content Marketing"]
    channels = ["Google Ads", "Facebook", "LinkedIn", "Email", "Twitter/X",
                "Instagram", "YouTube", "TikTok"]

    start_date = datetime(2023, 1, 1)

    records = []
    for i in range(n_rows):
        campaign = np.random.choice(campaign_types)
        channel = np.random.choice(channels)
        date = start_date + timedelta(days=np.random.randint(0, 730))

        impressions = int(np.random.lognormal(mean=10, sigma=1.2))
        impressions = min(impressions, 5_000_000)

        # CTR varies by channel
        base_ctr = {
            "Google Ads": 0.035, "Facebook": 0.028, "LinkedIn": 0.022,
            "Email": 0.045, "Twitter/X": 0.015, "Instagram": 0.032,
            "YouTube": 0.018, "TikTok": 0.025
        }[channel]
        ctr = round(max(0.001, np.random.normal(base_ctr, 0.01)), 4)

        clicks = max(1, int(impressions * ctr))
        conversion_rate = round(max(0.005, np.random.normal(0.035, 0.015)), 4)
        conversions = max(0, int(clicks * conversion_rate))

        spend = round(np.random.uniform(500, 50000), 2)
        revenue_generated = round(conversions * np.random.uniform(50, 500), 2)
        roi = round((revenue_generated - spend) / max(spend, 1) * 100, 2)

        cpc = round(spend / max(clicks, 1), 2)

        records.append({
            "date": date.strftime("%Y-%m-%d"),
            "campaign_type": campaign,
            "channel": channel,
            "impressions": impressions,
            "clicks": clicks,
            "ctr": ctr,
            "conversions": conversions,
            "conversion_rate": conversion_rate,
            "spend": spend,
            "revenue_generated": revenue_generated,
            "roi": roi,
            "cpc": cpc
        })

    df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    return df


def generate_customer_data(n_rows: int = 800, seed: int = 42) -> pd.DataFrame:
    """Generate realistic customer profile data."""
    np.random.seed(seed + 2)

    segments = ["Enterprise", "Mid-Market", "Small Business", "Startup", "Individual"]
    segment_weights = [0.15, 0.25, 0.30, 0.20, 0.10]

    industries = ["Technology", "Finance", "Healthcare", "Retail", "Manufacturing",
                  "Education", "Media", "Energy"]

    records = []
    for i in range(n_rows):
        segment = np.random.choice(segments, p=segment_weights)
        industry = np.random.choice(industries)

        # LTV varies by segment
        ltv_base = {
            "Enterprise": np.random.lognormal(10, 0.8),
            "Mid-Market": np.random.lognormal(9, 0.7),
            "Small Business": np.random.lognormal(8, 0.6),
            "Startup": np.random.lognormal(7.5, 0.9),
            "Individual": np.random.lognormal(6, 0.5)
        }[segment]

        lifetime_value = round(min(ltv_base, 500000), 2)
        age_months = max(1, int(np.random.exponential(24)))
        purchase_frequency = round(max(0.1, np.random.gamma(2, 1.5)), 1)
        satisfaction = round(np.clip(np.random.normal(7.2, 1.5), 1, 10), 1)
        support_tickets = max(0, int(np.random.poisson(3)))

        # Churn risk inversely related to satisfaction
        churn_base = (10 - satisfaction) / 10 * 0.6
        churn_risk = round(np.clip(churn_base + np.random.normal(0, 0.1), 0, 1), 2)

        nps_score = int(np.clip(np.random.normal(satisfaction * 10, 15), 0, 100))
        engagement_score = round(np.clip(np.random.beta(2, 3) * 100, 5, 100), 1)

        records.append({
            "customer_id": f"CUST-{i+1001:05d}",
            "segment": segment,
            "industry": industry,
            "lifetime_value": lifetime_value,
            "account_age_months": age_months,
            "purchase_frequency": purchase_frequency,
            "satisfaction_score": satisfaction,
            "support_tickets": support_tickets,
            "churn_risk": churn_risk,
            "nps_score": nps_score,
            "engagement_score": engagement_score
        })

    return pd.DataFrame(records)


def generate_github_data(n_rows: int = 600, seed: int = 42) -> pd.DataFrame:
    """Generate realistic GitHub repository data."""
    np.random.seed(seed + 3)

    languages = ["Python", "JavaScript", "TypeScript", "Rust", "Go",
                 "Java", "C++", "C#", "Ruby", "Swift", "Kotlin", "PHP"]
    language_weights = [0.22, 0.18, 0.14, 0.10, 0.09,
                        0.08, 0.06, 0.04, 0.03, 0.03, 0.02, 0.01]

    topics = ["web-framework", "machine-learning", "cli-tool", "database",
              "api-client", "devops", "data-science", "mobile", "security",
              "testing", "cloud", "blockchain"]

    adjectives = ["fast", "smart", "tiny", "super", "awesome", "hyper",
                  "micro", "nano", "ultra", "mega", "turbo", "auto"]
    nouns = ["api", "hub", "flow", "sync", "forge", "stack", "core",
             "kit", "lab", "vault", "pulse", "link", "engine", "craft"]

    records = []
    for i in range(n_rows):
        language = np.random.choice(languages, p=language_weights)
        topic = np.random.choice(topics)

        repo_name = f"{np.random.choice(adjectives)}-{np.random.choice(nouns)}"
        if np.random.random() > 0.5:
            repo_name += f"-{np.random.choice(['js', 'py', 'rs', 'go', 'io', 'ai'])}"

        stars = int(np.random.lognormal(5, 2))
        stars = min(stars, 200000)
        forks = max(0, int(stars * np.random.uniform(0.05, 0.4)))
        open_issues = max(0, int(np.random.poisson(stars * 0.02)))
        open_issues = min(open_issues, 5000)
        contributors = max(1, int(np.random.lognormal(2, 1.2)))
        contributors = min(contributors, 2000)

        days_ago = np.random.randint(0, 365)
        last_updated = (datetime(2024, 12, 31) - timedelta(days=days_ago)).strftime("%Y-%m-%d")

        # Code quality correlated with stars + contributors
        quality_base = min(10, (np.log1p(stars) + np.log1p(contributors)) / 3)
        code_quality = round(np.clip(quality_base + np.random.normal(0, 1), 1, 10), 1)

        license_types = ["MIT", "Apache-2.0", "GPL-3.0", "BSD-3-Clause", "ISC", "None"]
        repo_license = np.random.choice(license_types, p=[0.35, 0.25, 0.15, 0.10, 0.05, 0.10])

        has_ci = np.random.choice([True, False], p=[0.65, 0.35])
        has_docs = np.random.choice([True, False], p=[0.55, 0.45])

        records.append({
            "repo_name": repo_name,
            "language": language,
            "topic": topic,
            "stars": stars,
            "forks": forks,
            "open_issues": open_issues,
            "contributors": contributors,
            "last_updated": last_updated,
            "code_quality_score": code_quality,
            "license": repo_license,
            "has_ci_cd": has_ci,
            "has_documentation": has_docs
        })

    return pd.DataFrame(records)


def generate_all_datasets(output_dir: str = "data") -> dict:
    """Generate all datasets and save to CSV files."""
    os.makedirs(output_dir, exist_ok=True)

    datasets = {
        "sales_data": generate_sales_data(),
        "marketing_data": generate_marketing_data(),
        "customers_data": generate_customer_data(),
        "github_repos": generate_github_data()
    }

    for name, df in datasets.items():
        filepath = os.path.join(output_dir, f"{name}.csv")
        df.to_csv(filepath, index=False)
        print(f"✅ Generated {filepath}: {len(df)} rows, {len(df.columns)} columns")

    return datasets


if __name__ == "__main__":
    print("=" * 60)
    print("  Generating Datasets for AI Decision Platform")
    print("=" * 60)
    datasets = generate_all_datasets()
    print(f"\n✅ All {len(datasets)} datasets generated successfully!")
    for name, df in datasets.items():
        print(f"  📊 {name}: {df.shape}")
