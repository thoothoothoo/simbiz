import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

def create_pdf(df):
    buffer = BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    
    # Add title
    p.setFont("Helvetica-Bold", 16)
    p.drawString(100, 750, "Business Simulation Report")
    
    # Add summary
    p.setFont("Helvetica", 12)
    y_position = 700
    p.drawString(100, y_position, f"Final Bank Balance: ₹{df['Bank Balance'].iloc[-1]:,.0f}")
    y_position -= 20
    p.drawString(100, y_position, f"Total Revenue: ₹{df['Revenue'].sum():,.0f}")
    y_position -= 20
    p.drawString(100, y_position, f"Net Profit/Loss: ₹{df['Profit/Loss'].sum():,.0f}")
    
    # Add chart placeholders
    y_position -= 40
    p.drawString(100, y_position, "Cash Flow Chart:")
    y_position -= 120
    
    p.drawString(100, y_position, "Inventory Chart:")
    y_position -= 120
    
    p.drawString(100, y_position, "Cost Breakdown Chart:")
    
    p.showPage()
    p.save()
    buffer.seek(0)
    return buffer

def run_simulation():
    st.title("Advanced Business Simulation Tool")
    
    # Input parameters
    st.sidebar.header("Core Business Parameters")
    
    # Initial conditions
    bb = st.sidebar.number_input("Initial Bank Balance (₹)", min_value=0, value=100000, step=10000)
    
    # Manufacturing
    mp = st.sidebar.number_input("Manufacturing Cost per Product (₹)", min_value=0, value=50, step=5)
    initial_mpq = st.sidebar.number_input("Initial Production Quantity", min_value=0, value=1000, step=100)
    
    # Sales
    sp = st.sidebar.number_input("Selling Price per Product (₹)", min_value=0, value=100, step=5)
    max_payment_days = st.sidebar.number_input("Maximum Payment Days (d)", min_value=0, value=90, step=5)
    
    # Overhead
    co = st.sidebar.number_input("Monthly Overhead Cost (₹)", min_value=0, value=5000, step=500)
    
    # Capital injections
    capital_type = st.sidebar.radio("Capital Type", ["Equity", "Loan (30-day repayment)"])
    num_injections = st.sidebar.number_input("Number of Capital Injections", min_value=0, value=1, step=1)
    injections = []
    for i in range(num_injections):
        day = st.sidebar.number_input(f"Injection {i+1} - Day", min_value=0, value=30*(i+1))
        amount = st.sidebar.number_input(f"Injection {i+1} - Amount (₹)", min_value=0, value=50000)
        injections.append((day, amount))
    
    # Purchase patterns
    st.sidebar.header("Sales Parameters")
    purchase_pattern = st.sidebar.radio("Purchase Pattern", 
                                      ["Regular Intervals", "Random", "Specific Dates"])
    
    if purchase_pattern == "Regular Intervals":
        purchase_freq = st.sidebar.selectbox("Purchase Frequency", ["Weekly", "Bi-weekly", "Monthly"])
        pct_per_purchase = st.sidebar.slider("% of Inventory Purchased Each Time", 1, 100, 10)
    elif purchase_pattern == "Random":
        avg_purchase_days = st.sidebar.number_input("Average Days Between Purchases", min_value=1, value=7)
        pct_per_purchase = st.sidebar.slider("% of Inventory Purchased Each Time", 1, 100, 10)
    else:  # Specific Dates
        purchase_dates_input = st.sidebar.text_input("Enter Purchase Days (comma separated)", "15,30,45,60")
        purchase_dates = [int(x.strip()) for x in purchase_dates_input.split(",") if x.strip().isdigit()]
        pct_per_purchase = st.sidebar.slider("% of Inventory Purchased Each Time", 1, 100, 10)
    
    # Payment scenarios
    st.sidebar.header("Payment Scenarios")
    scenario = st.sidebar.selectbox("Select Payment Scenario", 
                                  ["Best Case (70% in 15 days)", 
                                   "Good Case (50% in 15 days)", 
                                   "Average Case (30% in 15 days)", 
                                   "Poor Case (20% in 30 days)", 
                                   "Worst Case (10% in 30 days)", 
                                   "Custom"])
    
    if scenario == "Custom":
        early_pct = st.sidebar.slider("% Paid in 15 days", 0, 100, 30)
        mid_pct = st.sidebar.slider("% Paid in 15-30 days", 0, 100, 50)
        late_pct = 100 - early_pct - mid_pct
    else:
        # Realistic payment scenarios centered around 30 days
        scenarios = {
            "Best Case (70% in 15 days)": [70, 25, 5],
            "Good Case (50% in 15 days)": [50, 40, 10],
            "Average Case (30% in 15 days)": [30, 50, 20],
            "Poor Case (20% in 30 days)": [20, 50, 30],
            "Worst Case (10% in 30 days)": [10, 40, 50]
        }
        early_pct, mid_pct, late_pct = scenarios[scenario]
    
    # Simulation period
    sim_months = st.sidebar.number_input("Simulation Period (months)", min_value=1, value=12, step=1)
    
    # Run simulation
    if st.sidebar.button("Run Simulation"):
        # Initialize data structures
        days = sim_months * 30  # Approximate
        date_range = pd.date_range(start=datetime.today(), periods=days, freq='D')
        
        # Initialize DataFrame
        df = pd.DataFrame(index=date_range, columns=[
            'Bank Balance', 'Revenue', 'Profit/Loss', 'Manufacturing Cost', 
            'Overhead Cost', 'Capital Injected', 'Loan Repayment',
            'Inventory', 'Products Sold', 'Outstanding Payments'
        ]).fillna(0)
        df.iloc[0, df.columns.get_loc('Bank Balance')] = bb
        df.iloc[0, df.columns.get_loc('Inventory')] = initial_mpq
        
        # Payment distribution based on scenario
        def create_payment_distribution(early, mid, late, max_days):
            # Early payments (0-15 days)
            early_days = min(15, max_days)
            early_dist = np.linspace(1, 0.5, early_days)
            early_dist = early_dist * (early/100) / early_dist.sum()
            
            # Mid payments (16-30 days)
            mid_days = min(15, max(0, max_days-15))
            mid_dist = np.linspace(0.8, 0.2, mid_days)
            mid_dist = mid_dist * (mid/100) / mid_dist.sum()
            
            # Late payments (31+ days)
            late_days = max(0, max_days-30)
            late_dist = np.linspace(0.5, 0.1, late_days) if late_days > 0 else np.array([])
            late_dist = late_dist * (late/100) / late_dist.sum() if late_days > 0 else np.array([])
            
            full_dist = np.concatenate([early_dist, mid_dist, late_dist])
            return full_dist / full_dist.sum()  # Normalize
        
        payment_dist = create_payment_distribution(early_pct, mid_pct, late_pct, max_payment_days)
        
        # Generate purchase days based on selected pattern
        if purchase_pattern == "Regular Intervals":
            freq_days = 7 if purchase_freq == "Weekly" else 14 if purchase_freq == "Bi-weekly" else 30
            purchase_days = [day for day in range(days) if day % freq_days == 0]
        elif purchase_pattern == "Random":
            purchase_days = []
            day = 0
            while day < days:
                purchase_days.append(day)
                day += np.random.poisson(avg_purchase_days)
        else:  # Specific Dates
            purchase_days = [day for day in purchase_dates if day < days]
        
       
        # Process each day
        loans = []  # Track loans: (day_taken, amount, due_day)
        
        for day in range(days):
            current_date = date_range[day]
            prev_row = df.iloc[day-1] if day > 0 else df.iloc[0]
            
            # Initialize today's values
            df.iloc[day] = prev_row.copy()
            df.iloc[day, df.columns.get_loc('Profit/Loss')] = 0  # Reset daily
            
            # Check for monthly overhead (1st of each month)
            if current_date.day == 1:
                df.iloc[day, df.columns.get_loc('Overhead Cost')] += co
                df.iloc[day, df.columns.get_loc('Bank Balance')] -= co
            
            # Check if this is a purchase day
            if day in purchase_days and df.iloc[day]['Inventory'] > 0:
                inventory = df.iloc[day]['Inventory']
                sell_qty = min(inventory, int(inventory * pct_per_purchase / 100))
                
                if sell_qty > 0:
                    df.iloc[day, df.columns.get_loc('Inventory')] -= sell_qty
                    df.iloc[day, df.columns.get_loc('Products Sold')] += sell_qty
                    
                    # Schedule payments
                    for payment_day in range(min(max_payment_days, days-day)):
                        payment_prob = payment_dist[payment_day] if payment_day < len(payment_dist) else 0
                        payment_amount = (sp * sell_qty) * payment_prob
                        df.iloc[day + payment_day, df.columns.get_loc('Revenue')] += payment_amount
                        df.iloc[day + payment_day, df.columns.get_loc('Profit/Loss')] += (sp - mp) * sell_qty * payment_prob
            
            # Check if inventory is empty and we have funds to manufacture
            if df.iloc[day]['Inventory'] <= 0 and (df.iloc[day]['Bank Balance'] - co) >= (mp * initial_mpq):
                manufacturing_cost = mp * initial_mpq
                df.iloc[day, df.columns.get_loc('Manufacturing Cost')] += manufacturing_cost
                df.iloc[day, df.columns.get_loc('Bank Balance')] -= manufacturing_cost
                df.iloc[day, df.columns.get_loc('Inventory')] += initial_mpq
            
            # Process capital injections
            for inj_day, amount in injections:
                if day == inj_day:
                    if capital_type == "Equity":
                        df.iloc[day, df.columns.get_loc('Capital Injected')] += amount
                        df.iloc[day, df.columns.get_loc('Bank Balance')] += amount
                    else:
                        # Loan - record for repayment
                        loans.append((day, amount, day + 30))
                        df.iloc[day, df.columns.get_loc('Capital Injected')] += amount
                        df.iloc[day, df.columns.get_loc('Bank Balance')] += amount
            
            # Process loan repayments
            for loan in loans[:]:
                taken_day, principal, due_day = loan
                if day == due_day:
                    if df.iloc[day]['Bank Balance'] >= principal:
                        df.iloc[day, df.columns.get_loc('Loan Repayment')] += principal
                        df.iloc[day, df.columns.get_loc('Bank Balance')] -= principal
                        loans.remove(loan)
            
            # Update bank balance with today's revenue
            df.iloc[day, df.columns.get_loc('Bank Balance')] += df.iloc[day]['Revenue']
        
        # Display results
        st.header("Simulation Results")
        
        # Summary statistics
        st.subheader("Key Metrics")
        final_balance = df.iloc[-1]['Bank Balance']
        total_revenue = df['Revenue'].sum()
        total_profit = df['Profit/Loss'].sum()
        total_manufacturing = df['Manufacturing Cost'].sum()
        total_overhead = df['Overhead Cost'].sum()
        total_capital = df['Capital Injected'].sum()
        total_loan_repayment = df['Loan Repayment'].sum()
        
        col1, col2 = st.columns(2)
        col1.metric("Final Bank Balance", f"₹{final_balance:,.0f}")
        col2.metric("Net Profit/Loss", f"₹{total_profit:,.0f}", 
                   delta_color="inverse" if total_profit < 0 else "normal")
        
        col1, col2 = st.columns(2)
        col1.metric("Total Revenue", f"₹{total_revenue:,.0f}")
        col2.metric("Total Costs", f"₹{total_manufacturing + total_overhead + total_loan_repayment:,.0f}")
        
        # Time period selection
        st.subheader("Detailed View")
        view_option = st.selectbox("View By", ["Days", "Weeks", "Months"])
        
        if view_option == "Weeks":
            view_df = df.resample('W').sum()
            view_df['Bank Balance'] = df['Bank Balance'].resample('W').last()
            view_df['Inventory'] = df['Inventory'].resample('W').last()
        elif view_option == "Months":
            view_df = df.resample('M').sum()
            view_df['Bank Balance'] = df['Bank Balance'].resample('M').last()
            view_df['Inventory'] = df['Inventory'].resample('M').last()
        else:
            view_df = df.copy()
        
        st.dataframe(view_df.style.format("{:,.0f}"))
        
        # Charts
        st.subheader("Performance Visualization")
        
        fig, ax = plt.subplots(3, 1, figsize=(12, 15))
        
        # Cash flow
        ax[0].plot(view_df.index, view_df['Bank Balance'], label='Bank Balance')
        ax[0].set_title("Cash Flow Over Time")
        ax[0].set_ylabel("Amount (₹)")
        ax[0].legend()
        
        # Inventory and Sales
        ax[1].bar(view_df.index, view_df['Inventory'], label='Inventory', alpha=0.6)
        ax[1].bar(view_df.index, view_df['Products Sold'], label='Products Sold', alpha=0.6)
        ax[1].set_title("Inventory Management")
        ax[1].legend()
        
        # Cost breakdown
        costs = view_df[['Manufacturing Cost', 'Overhead Cost', 'Loan Repayment']]
        ax[2].stackplot(view_df.index, 
                       costs['Manufacturing Cost'], 
                       costs['Overhead Cost'],
                       costs['Loan Repayment'],
                       labels=['Manufacturing', 'Overhead', 'Loan Repayment'])
        ax[2].set_title("Cost Breakdown")
        ax[2].legend()
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Export options
        st.subheader("Export Results")
        
        # Excel export
        excel_buffer = BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
            view_df.to_excel(writer, sheet_name='Simulation Results')
            writer.save()
        
        st.download_button(
            label="Download Excel",
            data=excel_buffer,
            file_name="business_simulation.xlsx",
            mime="application/vnd.ms-excel"
        )
        
        # PDF export
        pdf_buffer = create_pdf(view_df)
        st.download_button(
            label="Download PDF Report",
            data=pdf_buffer,
            file_name="business_simulation.pdf",
            mime="application/pdf"
        )

if __name__ == "__main__":
    run_simulation()
