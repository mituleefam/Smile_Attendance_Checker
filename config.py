# Page Configuration
st.set_page_config(page_title="Employee Dashboard", page_icon="🧑‍💼", layout="wide")
st.markdown("""
    <style>
        .employee-card {
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            margin-bottom: 15px;
            transition: all 0.3s ease;
        }
        .employee-card:hover {
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        .match-container {
            display: flex;
            justify-content: center;
            align-items: center;
            margin: 20px 0;
            gap: 20px;
        }
        .vector-space {
            background-color: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            margin-top: 20px;
        }
        
        .top-matches {
            display: grid;
            grid-template-columns: repeat(2, 1fr); /* Two columns */
            gap: 15px;
            padding: 10px 0;
        }

        .match-item {
            text-align: center;
        }
        .status-badge {
            padding: 5px 10px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 0.8em;
        }
        .checked-in {
            background-color: #d4edda;
            color: #155724;
            margin: 0 0 20px 0;
        }
        .not-checked {
            background-color: #f8d7da;
            color: #721c24;
            margin: 0 0 20px 0;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center;'>🧑‍💼 Face-Based Employee Check-in System</h1>", unsafe_allow_html=True)