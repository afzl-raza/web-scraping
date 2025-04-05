from flask import Flask, jsonify
import pandas as pd
import numpy as np
from flask_cors import CORS
from itertools import combinations
from collections import defaultdict
import traceback

app = Flask(__name__)
CORS(app)

def convert_to_native_types(obj):
    """Convert numpy/pandas types to native Python types for JSON serialization"""
    if isinstance(obj, (np.integer)):
        return int(obj)
    elif isinstance(obj, (np.floating)):
        return float(obj)
    elif isinstance(obj, (np.ndarray, pd.Series)):
        return obj.tolist()
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: convert_to_native_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_types(v) for v in obj]
    else:
        return obj

def load_data():
    try:
        # Load data with the correct column names
        df = pd.read_csv('cleaned_reduced.csv', parse_dates=['DateTime'])
        print("Data loaded successfully. Columns:", df.columns.tolist())
        print("Date range:", df['DateTime'].min(), "to", df['DateTime'].max())
        
        # Convert tags to list
        df['Tags'] = df['Tags'].apply(
            lambda x: [tag.strip().strip('"').strip("'") for tag in str(x).split(",")] if pd.notna(x) else []
        )
        
        # Explode tags for tag-based analysis
        df_exploded = df.explode('Tags')
        
        # Add temporal features
        df['Hour'] = df['DateTime'].dt.hour
        df['DayOfWeek'] = df['DateTime'].dt.dayofweek
        df['Month'] = df['DateTime'].dt.month
        df['Year'] = df['DateTime'].dt.year
        df['Date'] = df['DateTime'].dt.date
        
        return df, df_exploded
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        traceback.print_exc()
        return pd.DataFrame(), pd.DataFrame()

df, df_exploded = load_data()

@app.route('/api/tag_frequency')
def tag_frequency():
    try:
        if df_exploded.empty or 'Tags' not in df_exploded.columns:
            return jsonify({"success": True, "data": []})
            
        tag_counts = df_exploded['Tags'].value_counts().reset_index()
        tag_counts.columns = ['tag', 'count']
        tag_counts = tag_counts[tag_counts['tag'] != '']  # Remove empty tags
        
        return jsonify({
            "success": True,
            "data": convert_to_native_types(tag_counts.head(30).to_dict(orient='records'))
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/hourly_activity')
def hourly_activity():
    try:
        if df.empty or 'Hour' not in df.columns:
            return jsonify({"success": True, "data": []})
            
        hourly_activity = df.groupby('Hour').size().reset_index(name='count')
        return jsonify({
            "success": True,
            "data": convert_to_native_types(hourly_activity.to_dict(orient='records'))
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
@app.route('/api/daily_activity', methods=['GET'])
def daily_activity():
    try:
        # Check if data is loaded correctly
        if df.empty or 'DayOfWeek' not in df.columns:
            return jsonify({
                "success": True,
                "data": [],
                "message": "No data available or missing DayOfWeek column"
            })

        # Calculate activity per day of week
        daily_counts = df.groupby('DayOfWeek').size().reset_index(name='count')
        
        # Ensure all 7 days are represented (0=Monday to 6=Sunday)
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        complete_days = pd.DataFrame({
            'DayOfWeek': range(7),
            'DayName': day_names
        })
        
        # Merge with actual counts
        daily_counts = complete_days.merge(
            daily_counts, 
            on='DayOfWeek', 
            how='left'
        ).fillna(0)
        
        # Sort by day of week (0-6)
        daily_counts = daily_counts.sort_values('DayOfWeek')
        
        return jsonify({
            "success": True,
            "data": convert_to_native_types(daily_counts.to_dict(orient='records'))
        })
        
    except Exception as e:
        print(f"Error in daily_activity: {str(e)}")
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": "Internal server error",
            "data": []
        }), 500

@app.route('/api/time_trend')
def time_trend():
    try:
        if df.empty:
            return jsonify({
                "success": True,
                "data": [],
                "message": "No data available"
            })

        # Ensure DateTime column exists and is datetime type
        if 'DateTime' not in df.columns:
            return jsonify({
                "success": False,
                "error": "Required DateTime column not found"
            }), 400

        # Create monthly aggregates with proper error handling
        try:
            # Convert to monthly periods
            df['YearMonth'] = df['DateTime'].dt.to_period('M')
            
            # Group by month and count questions
            monthly_counts = df.groupby('YearMonth').size().reset_index(name='count')
            
            # Convert Period to timestamp and sort chronologically
            monthly_counts['Date'] = monthly_counts['YearMonth'].dt.to_timestamp()
            monthly_counts = monthly_counts.sort_values('Date')
            
            # Create complete date range to fill gaps
            min_date = monthly_counts['Date'].min().to_period('M')
            max_date = monthly_counts['Date'].max().to_period('M')
            all_months = pd.period_range(start=min_date, end=max_date, freq='M')
            
            # Reindex to include all months
            monthly_counts = monthly_counts.set_index('YearMonth').reindex(all_months, fill_value=0).reset_index()
            monthly_counts['Date'] = monthly_counts['index'].dt.to_timestamp()
            
            # Apply 3-month moving average
            monthly_counts['smoothed'] = monthly_counts['count'].rolling(
                window=3, 
                min_periods=1, 
                center=True
            ).mean().round(1)
            
            # Format for JSON
            result = monthly_counts[['Date', 'count', 'smoothed']].copy()
            result['Date'] = result['Date'].dt.strftime('%Y-%m')
            
            return jsonify({
                "success": True,
                "data": convert_to_native_types(result.to_dict(orient='records'))
            })
            
        except Exception as e:
            print(f"Error processing time trend: {str(e)}")
            traceback.print_exc()
            return jsonify({
                "success": False,
                "error": "Error processing time series data"
            }), 500

    except Exception as e:
        print(f"Unexpected error in time_trend endpoint: {str(e)}")
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": "Internal server error"
        }), 500

@app.route('/api/tag_trends')
def tag_trends():
    try:
        if df_exploded.empty or 'Tags' not in df_exploded.columns:
            return jsonify({"success": True, "data": [], "top_tags": []})

        # Get top 5 non-empty tags
        top_tags = df_exploded[df_exploded['Tags'] != '']['Tags'].value_counts().head(5).index.tolist()
        
        if not top_tags:
            return jsonify({"success": True, "data": [], "top_tags": []})

        # Filter for top tags
        df_top = df_exploded[df_exploded['Tags'].isin(top_tags)].copy()
        df_top['Year'] = df_top['DateTime'].dt.year
        df_top['Month'] = df_top['DateTime'].dt.month
        
        # Calculate total questions per month for normalization
        monthly_totals = df.groupby(['Year', 'Month']).size().reset_index(name='total_questions')
        
        # Calculate tag counts per month
        tag_counts = df_top.groupby(['Year', 'Month', 'Tags']).size().reset_index(name='count')
        
        # Merge with totals and calculate normalized percentage
        trends = pd.merge(tag_counts, monthly_totals, on=['Year', 'Month'])
        trends['percentage'] = (trends['count'] / trends['total_questions']) * 100
        trends['MonthYear'] = trends['Year'].astype(str) + '-' + trends['Month'].astype(str).str.zfill(2)
        
        # Fill missing months with 0%
        all_months = pd.date_range(
            start=f"{trends['Year'].min()}-{trends['Month'].min()}-01",
            end=f"{trends['Year'].max()}-{trends['Month'].max()}-01",
            freq='MS'
        ).to_period('M')
        
        complete_data = []
        for tag in top_tags:
            for month in all_months:
                year = month.year
                month_num = month.month
                match = trends[(trends['Tags'] == tag) & 
                              (trends['Year'] == year) & 
                              (trends['Month'] == month_num)]
                percentage = match['percentage'].values[0] if not match.empty else 0
                complete_data.append({
                    'Year': year,
                    'Month': month_num,
                    'Tags': tag,
                    'percentage': percentage,
                    'MonthYear': f"{year}-{str(month_num).zfill(2)}"
                })
        
        return jsonify({
            "success": True,
            "data": convert_to_native_types(complete_data),
            "top_tags": convert_to_native_types(top_tags)
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
@app.route('/api/tag_network', methods=['GET'])
def tag_network():
    try:
        if df.empty or 'Tags' not in df.columns:
            return jsonify({
                "success": True,
                "nodes": [],
                "edges": []
            })

        # Calculate co-occurrences
        cooccurrence = defaultdict(int)
        for tags in df['Tags']:
            tags = [t for t in tags if t]  # Remove empty tags
            if len(tags) >= 2:
                for pair in combinations(sorted(set(tags)), 2):
                    cooccurrence[pair] += 1

        # Get top pairs
        top_pairs = sorted(cooccurrence.items(), key=lambda x: -x[1])[:50]
        
        # Prepare nodes and edges
        nodes = []
        edges = []
        tag_counts = df_exploded['Tags'].value_counts().to_dict()
        
        for pair, count in top_pairs:
            tag1, tag2 = pair
            if tag1 not in [n['id'] for n in nodes]:
                nodes.append({
                    "id": tag1,
                    "label": tag1,
                    "value": tag_counts.get(tag1, 1)
                })
            if tag2 not in [n['id'] for n in nodes]:
                nodes.append({
                    "id": tag2,
                    "label": tag2,
                    "value": tag_counts.get(tag2, 1)
                })
            edges.append({
                "from": tag1,
                "to": tag2,
                "value": count
            })

        return jsonify({
            "success": True,
            "nodes": nodes,
            "edges": edges
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "nodes": [],
            "edges": []
        }), 500
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)