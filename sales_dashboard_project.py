"""
SALES PERFORMANCE DASHBOARD & FORECASTING SYSTEM
=================================================
Author: Your Name
Date: January 2026

Project Description:
This project analyzes sales data to identify business trends, top-performing 
products, and forecast future revenue. It demonstrates data cleaning, EDA,
visualization, and machine learning skills.

Key Techniques:
- Data manipulation with Pandas
- Statistical analysis and aggregation
- Data visualization with Matplotlib/Seaborn
- Time series forecasting with Linear Regression
- Customer segmentation analysis

Libraries Required:
- pandas: Data manipulation and analysis
- numpy: Numerical computations
- matplotlib: Data visualization
- seaborn: Statistical visualizations
- scikit-learn: Machine learning algorithms
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Configure visualization settings
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

class SalesAnalyzer:
    """
    A comprehensive sales analysis class that handles data generation,
    cleaning, analysis, visualization, and forecasting.
    
    INTERVIEW TIP: Using classes shows object-oriented programming skills
    and makes code more organized and reusable.
    """
    
    def __init__(self, n_records=5000, random_seed=42):
        """
        Initialize the analyzer with sample data.
        
        Args:
            n_records (int): Number of sales records to generate
            random_seed (int): Seed for reproducibility
            
        INTERVIEW TIP: Always set random_seed for reproducible results.
        Interviewers appreciate consistency in outputs.
        """
        self.n_records = n_records
        self.random_seed = random_seed
        np.random.seed(random_seed)
        self.df = None
        self.model = None
        
    def generate_sales_data(self):
        """
        Generate realistic sales data with proper distributions.
        
        INTERVIEW EXPLANATION:
        - Used datetime for proper date handling
        - Price varies by category (realistic business logic)
        - Added multiple dimensions: category, region, customer
        - This simulates real-world data complexity
        """
        print("=" * 70)
        print("SALES PERFORMANCE DASHBOARD & FORECASTING SYSTEM")
        print("=" * 70)
        print("\n[STEP 1] Generating Sample Sales Data...")
        
        # Date range: Last 2 years
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2025, 12, 31)
        date_range = (end_date - start_date).days
        
        # Business logic: More sales in recent months (seasonal trend)
        # INTERVIEW TIP: Adding trends makes data more realistic
        dates = []
        for _ in range(self.n_records):
            # Weighted towards recent dates (business growth)
            days_offset = int(np.random.triangular(0, date_range, date_range))
            dates.append(start_date + timedelta(days=days_offset))
        
        # Product catalog with realistic pricing
        categories = ['Electronics', 'Clothing', 'Home & Kitchen', 'Books', 'Sports']
        products = {
            'Electronics': ['Laptop', 'Smartphone', 'Headphones', 'Tablet', 'Smartwatch'],
            'Clothing': ['T-Shirt', 'Jeans', 'Jacket', 'Shoes', 'Dress'],
            'Home & Kitchen': ['Blender', 'Coffee Maker', 'Vacuum', 'Cookware', 'Air Fryer'],
            'Books': ['Fiction', 'Non-Fiction', 'Comics', 'Textbook', 'Biography'],
            'Sports': ['Yoga Mat', 'Dumbbells', 'Running Shoes', 'Bicycle', 'Tennis Racket']
        }
        
        # Price ranges by category (in INR)
        # INTERVIEW TIP: Dictionary comprehension for clean code
        price_ranges = {
            'Electronics': (5000, 80000),
            'Clothing': (500, 5000),
            'Home & Kitchen': (1000, 15000),
            'Books': (200, 2000),
            'Sports': (500, 10000)
        }
        
        # Generate records efficiently
        # INTERVIEW TIP: List comprehension is faster than loops for large data
        data = []
        customer_pool = [f"CUST{i:04d}" for i in range(1000, 3000)]  # 2000 unique customers
        regions = ['North', 'South', 'East', 'West', 'Central']
        
        for i in range(self.n_records):
            category = np.random.choice(categories, p=[0.30, 0.20, 0.20, 0.15, 0.15])  # Weighted selection
            product = np.random.choice(products[category])
            price_min, price_max = price_ranges[category]
            price = np.random.uniform(price_min, price_max)
            
            # Quantity: Most orders are small (1-2 items)
            quantity = np.random.choice([1, 2, 3, 4, 5], p=[0.5, 0.3, 0.1, 0.06, 0.04])
            revenue = price * quantity
            
            data.append({
                'Date': dates[i],
                'Order_ID': f"ORD{i+1:05d}",
                'Customer_ID': np.random.choice(customer_pool),
                'Category': category,
                'Product': product,
                'Quantity': quantity,
                'Price': round(price, 2),
                'Revenue': round(revenue, 2),
                'Region': np.random.choice(regions)
            })
        
        # Create DataFrame and sort by date (chronological order)
        self.df = pd.DataFrame(data).sort_values('Date').reset_index(drop=True)
        
        # Save to CSV
        self.df.to_csv('sales_data.csv', index=False)
        
        print(f"✓ Generated {len(self.df):,} sales records")
        print(f"✓ Date range: {self.df['Date'].min().date()} to {self.df['Date'].max().date()}")
        print(f"✓ Total revenue: ₹{self.df['Revenue'].sum():,.2f}")
        print(f"✓ Unique customers: {self.df['Customer_ID'].nunique():,}")
        print(f"✓ Saved to: sales_data.csv")
        
        return self.df
    
    def clean_and_prepare_data(self):
        """
        Clean data and create useful derived features.
        
        INTERVIEW EXPLANATION:
        - Check for nulls, duplicates, outliers
        - Create time-based features (year, month, quarter)
        - Feature engineering improves analysis quality
        """
        print("\n[STEP 2] Data Cleaning & Feature Engineering...")
        
        # Check data quality
        print(f"\nDataset Shape: {self.df.shape}")
        print(f"Memory Usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Missing values check
        missing = self.df.isnull().sum()
        if missing.sum() == 0:
            print("✓ No missing values found")
        else:
            print(f"⚠ Missing values:\n{missing[missing > 0]}")
        
        # Duplicate check
        duplicates = self.df.duplicated().sum()
        if duplicates == 0:
            print("✓ No duplicate records found")
        else:
            print(f"⚠ Found {duplicates} duplicates")
            self.df = self.df.drop_duplicates()
        
        # Feature engineering: Extract time components
        # INTERVIEW TIP: Time-based features are crucial for trend analysis
        self.df['Year'] = self.df['Date'].dt.year
        self.df['Month'] = self.df['Date'].dt.month
        self.df['Month_Name'] = self.df['Date'].dt.strftime('%B')
        self.df['Quarter'] = self.df['Date'].dt.quarter
        self.df['Day_of_Week'] = self.df['Date'].dt.day_name()
        self.df['Week'] = self.df['Date'].dt.isocalendar().week
        
        # Outlier detection using IQR method
        # INTERVIEW TIP: Always check for outliers in financial data
        Q1 = self.df['Revenue'].quantile(0.25)
        Q3 = self.df['Revenue'].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((self.df['Revenue'] < (Q1 - 1.5 * IQR)) | 
                    (self.df['Revenue'] > (Q3 + 1.5 * IQR))).sum()
        print(f"\n✓ Created 6 new time-based features")
        print(f"✓ Detected {outliers} potential outliers ({outliers/len(self.df)*100:.2f}%)")
        
        # Summary statistics
        print(f"\n{'-'*70}")
        print("SUMMARY STATISTICS")
        print(f"{'-'*70}")
        print(self.df[['Quantity', 'Price', 'Revenue']].describe().round(2))
        
        return self.df
    
    def perform_business_analysis(self):
        """
        Conduct comprehensive business intelligence analysis.
        
        INTERVIEW EXPLANATION:
        - Use groupby for aggregations (essential Pandas skill)
        - Calculate KPIs: total revenue, average order value, etc.
        - Identify top performers and trends
        """
        print(f"\n{'-'*70}")
        print("KEY BUSINESS INSIGHTS")
        print(f"{'-'*70}")
        
        # KPI 1: Total Revenue
        total_revenue = self.df['Revenue'].sum()
        print(f"\n1. TOTAL REVENUE: ₹{total_revenue:,.2f}")
        
        # KPI 2: Average Order Value (AOV)
        avg_order_value = self.df['Revenue'].mean()
        print(f"2. AVERAGE ORDER VALUE: ₹{avg_order_value:,.2f}")
        
        # KPI 3: Revenue by Category
        # INTERVIEW TIP: groupby + agg is core data analysis pattern
        print("\n3. REVENUE BY CATEGORY:")
        category_stats = self.df.groupby('Category').agg({
            'Revenue': ['sum', 'mean', 'count']
        }).round(2)
        category_stats.columns = ['Total_Revenue', 'Avg_Revenue', 'Order_Count']
        category_stats = category_stats.sort_values('Total_Revenue', ascending=False)
        
        for cat in category_stats.index:
            total = category_stats.loc[cat, 'Total_Revenue']
            avg = category_stats.loc[cat, 'Avg_Revenue']
            count = category_stats.loc[cat, 'Order_Count']
            pct = (total / total_revenue) * 100
            print(f"   {cat:.<20} ₹{total:>12,.0f} ({pct:>5.1f}%) | "
                  f"Avg: ₹{avg:>8,.0f} | Orders: {count:>4.0f}")
        
        # KPI 4: Top Products
        print("\n4. TOP 10 PRODUCTS BY REVENUE:")
        top_products = (self.df.groupby('Product')['Revenue']
                       .sum()
                       .sort_values(ascending=False)
                       .head(10))
        for idx, (prod, rev) in enumerate(top_products.items(), 1):
            print(f"   {idx:2d}. {prod:.<25} ₹{rev:>12,.0f}")
        
        # KPI 5: Regional Performance
        print("\n5. REVENUE BY REGION:")
        region_stats = self.df.groupby('Region')['Revenue'].agg(['sum', 'mean']).round(2)
        region_stats = region_stats.sort_values('sum', ascending=False)
        for region in region_stats.index:
            total = region_stats.loc[region, 'sum']
            avg = region_stats.loc[region, 'mean']
            print(f"   {region:.<20} ₹{total:>12,.0f} | Avg: ₹{avg:>8,.0f}")
        
        # KPI 6: Time-based trends
        print("\n6. MONTHLY TRENDS:")
        monthly = self.df.groupby(self.df['Date'].dt.to_period('M'))['Revenue'].sum()
        print(f"   Average Monthly Revenue: ₹{monthly.mean():,.2f}")
        print(f"   Best Month: {monthly.idxmax()} (₹{monthly.max():,.2f})")
        print(f"   Worst Month: {monthly.idxmin()} (₹{monthly.min():,.2f})")
        print(f"   Revenue Growth: {((monthly.iloc[-1] / monthly.iloc[0]) - 1) * 100:+.1f}%")
        
        # KPI 7: Customer Analysis
        print("\n7. CUSTOMER INSIGHTS:")
        customer_stats = self.df.groupby('Customer_ID')['Revenue'].agg(['sum', 'count'])
        print(f"   Total Unique Customers: {len(customer_stats):,}")
        print(f"   Avg Customer Lifetime Value: ₹{customer_stats['sum'].mean():,.2f}")
        print(f"   Avg Orders per Customer: {customer_stats['count'].mean():.1f}")
        
        # Pareto Analysis (80-20 rule)
        customer_revenue = customer_stats['sum'].sort_values(ascending=False)
        cumsum_pct = (customer_revenue.cumsum() / customer_revenue.sum() * 100)
        top_20_pct_idx = int(len(customer_revenue) * 0.2)
        revenue_from_top_20 = cumsum_pct.iloc[top_20_pct_idx]
        print(f"   Pareto: Top 20% customers → {revenue_from_top_20:.1f}% of revenue")
        
        return {
            'total_revenue': total_revenue,
            'category_stats': category_stats,
            'top_products': top_products,
            'region_stats': region_stats,
            'monthly_trends': monthly
        }
    
    def create_visualizations(self):
        """
        Create comprehensive dashboard visualizations.
        
        INTERVIEW EXPLANATION:
        - Used subplots for organized multi-chart dashboard
        - Different chart types for different data (bar, line, pie)
        - Color coding and formatting for professional look
        - Always add titles, labels, and legends
        """
        print(f"\n[STEP 4] Creating Visualizations...")
        
        # Create figure with 6 subplots
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        fig.suptitle('Sales Performance Dashboard', fontsize=18, fontweight='bold', y=0.995)
        
        # 1. Monthly Revenue Trend (Line Chart)
        # INTERVIEW TIP: Time series always use line charts
        ax1 = axes[0, 0]
        monthly_data = self.df.groupby(self.df['Date'].dt.to_period('M'))['Revenue'].sum()
        monthly_data_dates = monthly_data.index.to_timestamp()
        ax1.plot(monthly_data_dates, monthly_data.values, marker='o', 
                linewidth=2.5, markersize=6, color='#2ecc71')
        ax1.fill_between(monthly_data_dates, monthly_data.values, alpha=0.3, color='#2ecc71')
        ax1.set_title('Monthly Revenue Trend', fontsize=13, fontweight='bold', pad=10)
        ax1.set_xlabel('Month', fontsize=11)
        ax1.set_ylabel('Revenue (₹)', fontsize=11)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Category Performance (Bar Chart)
        ax2 = axes[0, 1]
        category_revenue = self.df.groupby('Category')['Revenue'].sum().sort_values(ascending=False)
        bars = ax2.bar(category_revenue.index, category_revenue.values, color='#3498db', edgecolor='black')
        ax2.set_title('Revenue by Category', fontsize=13, fontweight='bold', pad=10)
        ax2.set_xlabel('Category', fontsize=11)
        ax2.set_ylabel('Revenue (₹)', fontsize=11)
        ax2.tick_params(axis='x', rotation=45)
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'₹{height/1e6:.1f}M', ha='center', va='bottom', fontsize=9)
        
        # 3. Regional Distribution (Pie Chart)
        ax3 = axes[1, 0]
        region_revenue = self.df.groupby('Region')['Revenue'].sum()
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
        wedges, texts, autotexts = ax3.pie(region_revenue.values, labels=region_revenue.index,
                                            autopct='%1.1f%%', startangle=90, colors=colors)
        ax3.set_title('Revenue Distribution by Region', fontsize=13, fontweight='bold', pad=10)
        # Make percentage text bold
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        # 4. Top 10 Products (Horizontal Bar)
        ax4 = axes[1, 1]
        top_products = self.df.groupby('Product')['Revenue'].sum().sort_values(ascending=True).tail(10)
        bars = ax4.barh(top_products.index, top_products.values, color='#e74c3c', edgecolor='black')
        ax4.set_title('Top 10 Products by Revenue', fontsize=13, fontweight='bold', pad=10)
        ax4.set_xlabel('Revenue (₹)', fontsize=11)
        ax4.grid(True, alpha=0.3, axis='x', linestyle='--')
        
        # 5. Quarterly Comparison (Grouped Bar)
        ax5 = axes[2, 0]
        quarterly = self.df.groupby(['Year', 'Quarter'])['Revenue'].sum().unstack(fill_value=0)
        quarterly.plot(kind='bar', ax=ax5, width=0.8)
        ax5.set_title('Quarterly Revenue Comparison', fontsize=13, fontweight='bold', pad=10)
        ax5.set_xlabel('Year', fontsize=11)
        ax5.set_ylabel('Revenue (₹)', fontsize=11)
        ax5.legend(title='Quarter', title_fontsize=10, fontsize=9)
        ax5.tick_params(axis='x', rotation=0)
        ax5.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        # 6. Average Order Value by Category (Bar Chart)
        ax6 = axes[2, 1]
        category_avg = self.df.groupby('Category')['Revenue'].mean().sort_values(ascending=False)
        bars = ax6.bar(category_avg.index, category_avg.values, color='#9b59b6', edgecolor='black')
        ax6.set_title('Average Order Value by Category', fontsize=13, fontweight='bold', pad=10)
        ax6.set_xlabel('Category', fontsize=11)
        ax6.set_ylabel('Avg Revenue per Order (₹)', fontsize=11)
        ax6.tick_params(axis='x', rotation=45)
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'₹{height:,.0f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('sales_dashboard.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: sales_dashboard.png (6 visualizations)")
        plt.close()
        
    def build_forecasting_model(self):
        """
        Build and evaluate sales forecasting model using Linear Regression.
        
        INTERVIEW EXPLANATION:
        - Time series converted to supervised learning problem
        - 80-20 train-test split for validation
        - Linear Regression: simple but effective for trends
        - Evaluated using R² and MSE metrics
        - Made future predictions (next 3 months)
        
        INTERVIEW Q&A:
        Q: Why Linear Regression?
        A: Simple, interpretable, works well for linear trends. 
           For non-linear patterns, I'd use ARIMA or Prophet.
        
        Q: How to improve accuracy?
        A: Add features like seasonality, promotions, holidays.
           Try polynomial features or ensemble methods.
        """
        print(f"\n[STEP 5] Building Sales Forecasting Model...")
        
        # Aggregate to monthly level
        monthly_sales = (self.df.groupby(self.df['Date'].dt.to_period('M'))['Revenue']
                        .sum()
                        .reset_index())
        monthly_sales['Date'] = monthly_sales['Date'].dt.to_timestamp()
        monthly_sales['Month_Num'] = range(len(monthly_sales))
        
        # Feature: Month number (0, 1, 2, ...)
        # Target: Revenue
        X = monthly_sales[['Month_Num']].values
        y = monthly_sales['Revenue'].values
        
        # Train-test split (80-20)
        # INTERVIEW TIP: Always use train_test_split for proper evaluation
        split_idx = int(len(monthly_sales) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Train model
        self.model = LinearRegression()
        self.model.fit(X_train, y_train)
        
        # Get model parameters
        # INTERVIEW TIP: Understanding model coefficients is important
        slope = self.model.coef_[0]
        intercept = self.model.intercept_
        print(f"\nModel Equation: Revenue = {intercept:,.0f} + {slope:,.0f} × Month")
        print(f"Interpretation: Revenue increases by ₹{slope:,.0f} per month")
        
        # Make predictions
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        # Evaluate model
        # INTERVIEW TIP: Know what these metrics mean
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        test_mse = mean_squared_error(y_test, y_pred_test)
        test_rmse = np.sqrt(test_mse)
        
        print(f"\n{'Model Performance':^50}")
        print(f"{'-'*50}")
        print(f"Training R² Score:   {train_r2:.4f} ({train_r2*100:.2f}% variance explained)")
        print(f"Test R² Score:       {test_r2:.4f} ({test_r2*100:.2f}% variance explained)")
        print(f"Test RMSE:           ₹{test_rmse:,.2f}")
        print(f"Mean Actual Revenue: ₹{y_test.mean():,.2f}")
        print(f"Prediction Error:    {(test_rmse/y_test.mean())*100:.2f}%")
        
        # INTERVIEW TIP: Always validate model isn't overfitting
        if train_r2 - test_r2 > 0.1:
            print("⚠ Warning: Possible overfitting detected")
        else:
            print("✓ Model generalizes well (no overfitting)")
        
        # Forecast next 3 months
        last_month = len(monthly_sales)
        future_months = np.array([[last_month + i] for i in range(1, 4)])
        future_predictions = self.model.predict(future_months)
        
        print(f"\n{'Next 3 Months Forecast':^50}")
        print(f"{'-'*50}")
        for i, pred in enumerate(future_predictions, 1):
            print(f"Month +{i}:  ₹{pred:>15,.2f}")
        
        # Visualization
        plt.figure(figsize=(14, 7))
        
        # Plot training data
        plt.scatter(X_train, y_train, color='#3498db', s=100, alpha=0.6, 
                   label='Training Data', edgecolors='black', linewidth=1)
        plt.plot(X_train, y_pred_train, color='#3498db', linewidth=2.5, 
                linestyle='--', alpha=0.8)
        
        # Plot test data
        plt.scatter(X_test, y_test, color='#2ecc71', s=100, alpha=0.6, 
                   label='Test Data (Actual)', edgecolors='black', linewidth=1)
        plt.scatter(X_test, y_pred_test, color='#e74c3c', s=100, alpha=0.6, 
                   marker='s', label='Test Data (Predicted)', edgecolors='black', linewidth=1)
        
        # Plot forecast
        plt.scatter(future_months, future_predictions, color='#9b59b6', s=150, 
                   alpha=0.8, marker='D', label='Future Forecast', 
                   edgecolors='black', linewidth=1.5)
        
        # Regression line extended
        all_months = np.vstack([X, future_months])
        plt.plot(all_months, self.model.predict(all_months), 
                color='gray', linewidth=2, linestyle=':', alpha=0.5, label='Trend Line')
        
        plt.title('Sales Forecasting Model - Linear Regression', 
                 fontsize=16, fontweight='bold', pad=15)
        plt.xlabel('Month Number', fontsize=12, fontweight='bold')
        plt.ylabel('Revenue (₹)', fontsize=12, fontweight='bold')
        plt.legend(fontsize=10, loc='upper left')
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.savefig('sales_forecast.png', dpi=300, bbox_inches='tight')
        print("\n✓ Saved: sales_forecast.png")
        plt.close()
        
        return {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_rmse': test_rmse,
            'future_predictions': future_predictions
        }
    
    def generate_report(self):
        """Generate a comprehensive analysis report."""
        print(f"\n{'='*70}")
        print("PROJECT EXECUTION COMPLETE!")
        print(f"{'='*70}")
        print("\n📁 Generated Files:")
        print("   1. sales_data.csv          - Raw sales data (5000 records)")
        print("   2. sales_dashboard.png     - 6 visualization charts")
        print("   3. sales_forecast.png      - Forecasting model visualization")
        
        
        
    
        
        print(f"\n{'='*70}\n")


def main():
    
    # Initialize analyzer
    analyzer = SalesAnalyzer(n_records=5000, random_seed=42)
    
    # Execute analysis pipeline
    analyzer.generate_sales_data()
    analyzer.clean_and_prepare_data()
    analyzer.perform_business_analysis()
    analyzer.create_visualizations()
    analyzer.build_forecasting_model()
    analyzer.generate_report()
    
    

if __name__ == "__main__":
    main()