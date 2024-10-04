import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load data function
@st.cache_data
def load_data(file):
    return pd.read_excel(file)

# Function to process data
def process_data(df, selected_product):
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df = df[df['Product'] == selected_product]

    # Resampling to monthly data and summing the invoice amounts
    monthly_revenue = df.resample('M', on='InvoiceDate')['InvoiceAmount'].sum().reset_index()

    # Ensure we only consider the last 6 months
    end_date = monthly_revenue['InvoiceDate'].max()
    start_date = end_date - pd.DateOffset(months=6)
    monthly_revenue = monthly_revenue[(monthly_revenue['InvoiceDate'] >= start_date) & (monthly_revenue['InvoiceDate'] <= end_date)]

    return monthly_revenue

# Function to predict revenue
def predict_revenue(monthly_revenue):
    # Prepare the data for regression
    monthly_revenue['Month'] = np.arange(len(monthly_revenue)).reshape(-1, 1)

    # Using Polynomial Regression
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(monthly_revenue[['Month']])

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_poly, monthly_revenue['InvoiceAmount'], test_size=0.2, random_state=42)

    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict for the next 6 months
    future_months = np.arange(len(monthly_revenue), len(monthly_revenue) + 6).reshape(-1, 1)
    future_months_poly = poly.transform(future_months)
    predicted_revenue = model.predict(future_months_poly)

    # Create a DataFrame for future predictions
    future_dates = pd.date_range(start=monthly_revenue['InvoiceDate'].iloc[-1] + pd.DateOffset(months=1), periods=6, freq='M')
    future_revenue_df = pd.DataFrame({'Month': future_dates, 'Predicted Revenue': predicted_revenue})

    return future_revenue_df

# Function to calculate top customers
def top_customers(df, selected_product):
    df = df[df['Product'] == selected_product]
    top_customers = df.groupby('Customer')['InvoiceAmount'].sum().sort_values(ascending=False).head(10)
    return top_customers.reset_index()

# Streamlit app
def main():
    st.title("Product Sales Prediction")

    # File upload
    uploaded_file = st.file_uploader("Upload your Excel file", type="xlsx")

    if uploaded_file:
        df = load_data(uploaded_file)

        # Product selection (sorted)
        products = sorted(df['Product'].unique())
        selected_product = st.selectbox("Select a product", products)

        # Process data
        monthly_revenue = process_data(df, selected_product)

        # Display past revenue and predictions
        st.subheader("Past 6 Months Revenue")
        fig, ax = plt.subplots(figsize=(10, 5))

        # Combine past and future revenue for full trend line
        all_revenue = pd.concat([monthly_revenue, predict_revenue(monthly_revenue)])

        # Plot past revenue
        ax.plot(monthly_revenue['InvoiceDate'], monthly_revenue['InvoiceAmount'], marker='o', label='Past Revenue', color='blue')

        # Plot future revenue
        future_revenue = predict_revenue(monthly_revenue)
        ax.plot(future_revenue['Month'], future_revenue['Predicted Revenue'], marker='o', linestyle='--', label='Predicted Revenue', color='green')

        # Current date line
        current_date = pd.Timestamp.now()
        ax.axvline(x=current_date, color='red', linestyle='--', label='Current Date')

        # Set x-ticks to show only month and year
        all_dates = pd.date_range(start=monthly_revenue['InvoiceDate'].min(), end=future_revenue['Month'].max(), freq='M')
        ax.set_xticks(all_dates)
        ax.set_xticklabels(all_dates.strftime('%b %Y'), rotation=45)

        ax.set_xlabel("Date")
        ax.set_ylabel("Revenue")
        ax.set_title("Monthly Revenue Trend")
        ax.legend()
        st.pyplot(fig)

        # Display table of past 6 months revenue
        st.subheader("Table of Past 6 Months Revenue")
        st.write(monthly_revenue)

        # Display predictions
        future_revenue_df = predict_revenue(monthly_revenue)
        st.subheader("Predicted Revenue for Next 6 Months")
        st.write(future_revenue_df)

        # Button to show top customers
        if st.button("Show Top 10 Customers"):
            top_customers_df = top_customers(df, selected_product)
            st.subheader("Top 10 Customers")
            st.write(top_customers_df)

if __name__ == "__main__":
    main()
