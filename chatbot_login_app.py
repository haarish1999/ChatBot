import streamlit as st
import sqlite3
import bcrypt
import time
import pandas as pd
import sqlite3.dbapi2 as sqlite
from io import BytesIO
import json
from sqlite3 import Binary # <<< IMPORTANT: Ensure Binary is imported for saving BLOBs

# ==============================
# DATA UTILITY FUNCTIONS
# ==============================

def json_to_simple_entities(entities_json):
    """Converts a JSON string of entities (from DB) back to the simple k:v, k:v string for the UI."""
    if not entities_json: 
        return ""
    try:
        entities_dict = json.loads(entities_json)
        # Handle cases where entities_dict might be a list or non-dict due to bad JSON
        if isinstance(entities_dict, dict):
            return ", ".join(f"{k}:{v}" for k, v in entities_dict.items())
        else:
            return "" # Return empty string if not a dictionary
    except (json.JSONDecodeError, TypeError):
        return ""

# FILE: chatbot_login_app.py

def get_existing_annotation(user_email, workspace_name, sentence):
    """Retrieves existing intent and entities for a given sentence from the DB."""
    local_cursor = conn.cursor()
    local_cursor.execute(
        """SELECT intent, entities_json FROM annotations  
           WHERE user_email=? AND workspace_name=? AND sentence=?""",
        (user_email, workspace_name, sentence)
    )
    result = local_cursor.fetchone()
    # Returns (intent, entities_json_string) or (None, None)
    return result if result else (None, None)

def save_annotation_to_db(workspace_name, user_email, sentence, intent, entities_json):
    """Saves or updates the annotation using UPSERT (ON CONFLICT)."""
    try:
        local_cursor = conn.cursor()
        # The ON CONFLICT clause handles the UPDATE if the row already exists
        local_cursor.execute(
            """
            INSERT INTO annotations (workspace_name, user_email, sentence, intent, entities_json, last_modified) 
            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(workspace_name, user_email, sentence) DO UPDATE SET
            intent = excluded.intent, 
            entities_json = excluded.entities_json,
            last_modified = CURRENT_TIMESTAMP
            """, 
            (workspace_name, user_email, sentence, intent, entities_json)
        )
        conn.commit()
        return True
    except Exception as e:
        # Assuming st is defined globally
        st.error(f"Failed to save annotation to database: {e}") 
        return False

# ==============================
# PAGE CONFIGURATION
# ==============================
st.set_page_config(page_title="BuddyBot", page_icon="🤖", layout="wide")

# ==============================
# DATABASE SETUP (THREAD-SAFE CONNECTION & MIGRATION)
# ==============================
@st.cache_resource
def get_db_connection():
    """Creates and caches the SQLite connection object."""
    # Use check_same_thread=False for Streamlit's multithreaded environment
    conn = sqlite3.connect("users.db", check_same_thread=False)
    return conn

conn = get_db_connection()

# Initialize tables and handle schema migration
cursor = conn.cursor()

# 1. Users Table
cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        email TEXT UNIQUE,
        password TEXT
    )
""")

# 2. Workspaces Table
cursor.execute("""
    CREATE TABLE IF NOT EXISTS workspaces (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_email TEXT,
        workspace_name TEXT UNIQUE, 
        domain TEXT,
        last_modified TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
""")

# 3. Datasets Table (Stores original CSV BLOB)
cursor.execute("""
    CREATE TABLE IF NOT EXISTS datasets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        workspace_name TEXT, 
        user_email TEXT,
        filename TEXT,
        data BLOB
    )
""")

# FILE: chatbot_login_app.py

# ... (inside DATABASE SETUP section) ...

# 4. Annotations Table (Stores labeled data) - FIX: Added last_modified
cursor.execute("""
    CREATE TABLE IF NOT EXISTS annotations (
        user_email TEXT,
        workspace_name TEXT,
        sentence TEXT,
        intent TEXT,
        entities_json TEXT,  
        last_modified TIMESTAMP DEFAULT CURRENT_TIMESTAMP, 
        PRIMARY KEY (user_email, workspace_name, sentence)
    )
""")
# ...

# 5. Models Table (NEW: Stores metadata about trained models)
cursor.execute("""
    CREATE TABLE IF NOT EXISTS models (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        workspace_name TEXT UNIQUE,
        model_engine TEXT, -- e.g., 'spaCy', 'Rasa', 'HuggingFace'
        model_version TEXT,
        training_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
""")

# Commit all table creations/migrations
conn.commit()


# ==============================
# DOMAIN DATA & NLU INTENTS
# ==============================
DOMAINS = {
    "Sports": {
        "icon": "⚽",
        "description": "Analyze team stats, game history, and player performance.",
        "intents": ["request_score", "query_player_stat", "book_ticket", "greeting"]
    },
    "Education": {
        "icon": "📚",
        "description": "Create learning assistants from textbooks, notes, or research papers.",
        "intents": ["query_definition", "request_summary", "schedule_study", "greeting"]
    },
    "Art & Design": {
        "icon": "🎨",
        "description": "Interpret artistic styles, history, or design principles.",
        "intents": ["query_artist", "describe_style", "find_gallery", "greeting"]
    },
    "Entertainment": {
        "icon": "🎬",
        "description": "Discuss movies, music, celebrities, and pop culture trends.",
        "intents": ["recommend_movie", "query_actor", "buy_merch", "greeting"]
    },
    "Finance": {
        "icon": "💰",
        "description": "Manage bank account queries, fund transfers, and fraud reporting.",
        "intents": ["query_balance", "transfer_funds", "report_fraud", "greeting"]
    },
    "Travel & Booking": {
        "icon": "✈️",
        "description": "Handle flight, hotel, and general ticket reservations and inquiries.",
        "intents": ["book_flight", "check_inquiry", "cancel_reservation", "greeting"]
    },
    "Business": {
        "icon": "💼",
        "description": "Handle general customer FAQs, submit help desk tickets, or analyze operations.",
        "intents": ["query_hours", "submit_complaint", "request_report", "greeting"]
    },
    "Healthcare": {
        "icon": "🏥",
        "description": "Provide information on symptoms, book appointments, or answer medical FAQs.",
        "intents": ["book_appointment", "query_symptom", "refill_prescription", "greeting"]
    },
    "IT Support": {
        "icon": "💻",
        "description": "Assist with troubleshooting, password resets, and software installation guides.",
        "intents": ["reset_password", "troubleshoot_login", "request_software", "greeting"]
    },
    "Real Estate": {
        "icon": "🏠",
        "description": "Search property listings, schedule viewings, and answer mortgage questions.",
        "intents": ["search_property", "schedule_viewing", "query_mortgage", "greeting"]
    },
    "E-commerce": {
        "icon": "🛒",
        "description": "Track orders, process returns, and handle product inventory questions.",
        "intents": ["track_order", "process_return", "query_inventory", "greeting"]
    }
}

# ==============================
# PAGE STYLING (Embedded CSS)
# ==============================
st.markdown("""
    <style>
        /* Base Styling */
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(135deg, #0b1a37 0%, #1c2b4d 100%);
            color: white;
            animation: fadeIn 1s ease-in-out;
        }
        .title {
            text-align: center;
            font-size: 36px;
            font-weight: 800;
            color: #6EC6FF;
            margin-bottom: 20px;
        }
        
        /* Sidebar Bot Avatar */
        .sidebar-bot-avatar {
            text-align: center;
            margin-bottom: 20px;
        }
        .sidebar-bot-avatar img {
            width: 80px;
            height: 80px;
            border-radius: 50%;
            background-color: #6EC6FF;
            padding: 5px;
            box-shadow: 0 0 10px rgba(110, 198, 255, 0.5);
        }
        .sidebar-bot-avatar p {
            font-size: 18px;
            font-weight: bold;
            color: white;
            margin-top: 10px;
        }
            
        /* Logo & Chat Bubbles */
        .logo-container {
            text-align: center;
            margin-top: 30px;
            animation: slideUp 1s ease-in-out;
        }
        @keyframes slideUp {
            from { opacity: 0; transform: translateY(40px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .chat-bubble {
            background-color: #2b70f0;
            border-radius: 15px;
            padding: 10px 15px;
            display: inline-block;
            margin: 10px 0;
            color: white;
            max-width: 90%;
            text-align: left;
        }
        .chat-bubble-container {
            margin-top: 20px;
            width: 100%;
            text-align: center;
        }

        /* Input & Auth Button Styles */
        .stTextInput > div > div > input, .stSelectbox > div > div > div > input {
            background-color: #1c2b4d;
            border: 1px solid #334466;
            color: white;
            transition: all 0.2s ease;
        }
        .stTextInput > div > div > input:focus, .stTextInput > div > div > input:hover {
            border-color: #6EC6FF !important;
            box-shadow: 0 0 0 1px #6EC6FF;
        }
        div.stButton button[kind="primary"] {
            background-color: white !important;
            color: #2b70f0 !important;
            border: 1px solid #2b70f0;
            font-weight: bold;
            box-shadow: 0 4px 10px rgba(0,0,0,0.3);
            transition: all 0.3s ease-in-out;
        }
        div.stButton button[kind="primary"]:hover {
            background-color: #f0f0f0 !important;
            transform: scale(1.02);
            border-color: #1748b0;
        }
        
        /* Custom Styling for Simple Domain Cards/Buttons */
        .domain-card {
            background-color: #1c2b4d;
            border: 2px solid #334466;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 15px;
            transition: all 0.3s ease;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
            text-align: center;
        }
        .domain-card:hover {
            border-color: #6EC6FF;
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.5);
            transform: translateY(-3px);
        }
        .domain-card h3 {
            color: #6EC6FF;
            margin-top: 0;
            font-size: 20px;
        }
        .domain-card p {
            font-size: 14px;
            color: #ccc;
        }
        
        /* Sidebar styling for better appearance */
        [data-testid="stSidebar"] {
            background-color: #1c2b4d !important;
            color: white;
        }
        
        /* New: Entity Tagging Style Simulation */
        .sentence-display {
            font-size: 1.2em;
            padding: 15px;
            margin: 10px 0;
            background-color: #1c2b4d;
            border-radius: 8px;
            border: 1px solid #334466;
            min-height: 50px;
        }
    </style>
""", unsafe_allow_html=True)

# ==============================
# SESSION STATE & NAVIGATION HELPERS
# ==============================
if 'page' not in st.session_state:
    st.session_state.page = 'register'
if 'logged_in_email' not in st.session_state:
    st.session_state.logged_in_email = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'current_workspace' not in st.session_state:
    st.session_state.current_workspace = None
if 'current_domain' not in st.session_state:
    st.session_state.current_domain = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = {}
if 'temp_workspace_name' not in st.session_state:
    st.session_state.temp_workspace_name = ""
if 'workspace_action' not in st.session_state:
    st.session_state.workspace_action = None
if 'sentences_df' not in st.session_state: 
    st.session_state.sentences_df = None
if 'annotation_index' not in st.session_state: 
    st.session_state.annotation_index = 0

# Check query parameters for initial navigation
if 'page' in st.query_params:
    st.session_state.page = st.query_params['page']

# Navigation Functions
def navigate_to_login():
    st.session_state.page = 'login'
    st.query_params['page'] = 'login'

def navigate_to_register():
    st.session_state.page = 'register'
    st.query_params['page'] = 'register'

def navigate_to_home():
    st.session_state.page = 'home'
    st.query_params['page'] = 'home'

def navigate_to_create_workspace():
    st.session_state.page = 'create_workspace'
    st.query_params['page'] = 'create_workspace'

def navigate_to_workspace():
    st.session_state.page = 'workspace'
    st.query_params['page'] = 'workspace'

def navigate_to_action_choice():
    st.session_state.page = 'action_choice'
    st.query_params['page'] = 'action_choice'
    
def navigate_to_annotate(): 
    st.session_state.page = 'annotate'
    st.query_params['page'] = 'annotate'

def navigate_to_policy():
    st.session_state.page = 'policy'
    st.query_params['page'] = 'policy'
    
# Callback function for domain selection
def finalize_workspace_creation(workspace_name, domain_name):
    """Callback for creating and activating a new workspace."""
    local_cursor = conn.cursor() 
    
    try:
        local_cursor.execute(
            "INSERT INTO workspaces (user_email, workspace_name, domain) VALUES (?, ?, ?)",
            (st.session_state.logged_in_email, workspace_name, domain_name)
        )
        conn.commit()
        
        st.session_state.current_workspace = workspace_name
        st.session_state.current_domain = domain_name
        st.session_state.messages.clear()
        st.session_state.temp_workspace_name = "" 
        
        navigate_to_action_choice()
    except sqlite3.IntegrityError:
        st.error("⚠️ A workspace with that name already exists. Please choose a different name.")
        
# Callback for activating existing workspace
def activate_existing_workspace(workspace_name, domain_name):
    """Callback for setting an existing workspace as active."""
    st.session_state.current_workspace = workspace_name
    st.session_state.current_domain = domain_name
    
    if workspace_name not in st.session_state.chat_history:
        st.session_state.chat_history[workspace_name] = []
        
    st.session_state.messages = st.session_state.chat_history.get(workspace_name, [])
    
    navigate_to_action_choice()

# Callback to set action and move to workspace page
def set_workspace_action(action):
    """Callback for setting the current action (Train, Test, Annotate, Evaluate)."""
    st.session_state.workspace_action = action
    
    if action == "Annotate":
        navigate_to_annotate()
    else:
        navigate_to_workspace()

# ==============================
# SIDEBAR CONTENT FUNCTION
# ==============================
def show_sidebar_content():
    # Show sidebar on ALL pages EXCEPT the 'register' page
    if st.session_state.page == 'register':
        return

    user_email = st.session_state.logged_in_email
    workspace = st.session_state.current_workspace 
    
    st.sidebar.markdown("""
        <div class="sidebar-bot-avatar">
            <img src="https://cdn-icons-png.flaticon.com/512/4712/4712100.png" alt="Bot Avatar">
            <p>BuddyBot</p>
        </div>
    """, unsafe_allow_html=True)
    st.sidebar.markdown("---")
    
    # Check if the user is logged in to display user/workspace specific details
    if user_email:
        st.sidebar.markdown(f"**Current Workspace:** `{workspace}`")
        st.sidebar.markdown(f"**User:** `{user_email}`")
        
        # Logout button 
        st.sidebar.button("Logout", key="logout_sidebar", use_container_width=True, on_click=lambda: (
            st.session_state.pop('logged_in_email', None), 
            st.session_state.pop('current_workspace', None), 
            st.session_state.pop('current_domain', None), 
            st.session_state.messages.clear(), 
            st.session_state.chat_history.clear(),
            navigate_to_login()
        ))
        st.sidebar.markdown("---")

        with st.sidebar.expander("❓ **Help Section**"):
            st.markdown("1. **Select/Create Workspace** to specialize the bot.")
            st.markdown("2. **Upload a CSV** to train the bot.")
            st.markdown("3. **Annotate** the data to teach the NLU model.") 
            st.markdown("4. **Train** the model then **Chat**.") 
        st.sidebar.markdown("---")

        if workspace:
            st.sidebar.markdown(f"**Chat History: {workspace}**") 
            if st.session_state.chat_history.get(workspace):
                with st.sidebar.container(height=200):
                    for i, msg in enumerate(st.session_state.chat_history[workspace]):
                        if msg["role"] == "user":
                            summary = msg["content"][:30] + "..." if len(msg["content"]) > 30 else msg["content"]
                            st.sidebar.markdown(f"*{i+1}. {summary}*")
            else:
                st.sidebar.markdown("No history for this workspace yet.")
        else:
            st.sidebar.markdown("**Activate a workspace to view chat history.**")
        st.sidebar.markdown("---")

    else:
        # Simple message for non-logged-in users on login/policy pages
        st.sidebar.markdown("""
            <p style='color:#ccc; text-align:center;'>Please log in to access your workspaces and features.</p>
        """, unsafe_allow_html=True)

# ==============================
# DATA LOADERS/HANDLERS
# ==============================

@st.cache_data
def load_dataset_blob(user_email, workspace_name):
    """Retrieves the dataset BLOB and converts it to a DataFrame."""
    local_cursor = conn.cursor()
    local_cursor.execute(
        "SELECT data FROM datasets WHERE user_email=? AND workspace_name=?", 
        (user_email, workspace_name)
    )
    result = local_cursor.fetchone()
    
    if result:
        data_blob = result[0]
        try:
            # Convert BLOB to BytesIO object, then read as CSV
            df = pd.read_csv(BytesIO(data_blob))
            return df
        except Exception as e:
            # Print error to console/logs for better debugging if loading fails
            print(f"Error reading data from DB: {e}") 
            st.error("Error reading data from DB. File format might be corrupted.")
            return None
    return None

def split_dataframe_to_sentences(df):
    """
    Splits the content of the primary text column into a list of sentences.
    """
    text_col = None
    
    # 1. Search for a preferred column name (case-insensitive for robustness)
    for col in df.columns:
        if col.lower() in ['text', 'sentence', 'utterance']:
            text_col = col
            break
    
    if text_col:
        # Simple splitting for demonstration
        # Ensure only string data is used for splitting
        all_text = " ".join(df[text_col].astype(str).tolist())
        # Basic sentence splitting by period
        sentences = [s.strip() for s in all_text.split('.') if s.strip()]
        return pd.DataFrame(sentences, columns=['sentence'])
    
    st.warning("Could not find a 'text', 'sentence', or 'utterance' column in the uploaded dataset. Using all columns as one string.")
    
    # Fallback: concatenate all columns for a simple string split
    all_text = df.astype(str).agg(' '.join, axis=1).str.cat(sep=' ')
    sentences = [s.strip() for s in all_text.split('.') if s.strip()]
    return pd.DataFrame(sentences, columns=['sentence'])


# ==============================
# NLU MODEL INTEGRATION (STUBS)
# ==============================

def train_nlu_model(workspace_name, annotated_data):
    """
    Placeholder function for NLU model training.
    """
    # Check for actual data to prevent empty training
    if annotated_data is None or annotated_data.empty or len(annotated_data) == 0:
        st.error("Cannot train: No annotated data found in the database for this workspace.")
        return False

    # 1. Simulate Training Time
    with st.spinner("⏳ Training NLU Model..."):
        time.sleep(3) 

    # 2. Simulate Model Saving/Metadata Update
    local_cursor = conn.cursor()
    
    # Store/Update model metadata
    local_cursor.execute("DELETE FROM models WHERE workspace_name=?", (workspace_name,))
    local_cursor.execute(
        "INSERT INTO models (workspace_name, model_engine, model_version) VALUES (?, ?, ?)",
        (workspace_name, "spaCy_Simulated", "v1.0")
    )
    conn.commit()

    st.success(f"✅ NLU Model trained and saved! Engine: spaCy_Simulated, Version: v1.0. Now ready to **Test**.")
    return True

# ==============================
# CHAT LOGIC (UPDATED WITH SIMULATED NLU)
# ==============================
def display_chat_messages():
    """Displays all messages stored in the current session state chat history."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def predict_intent_and_entities(prompt, domain):
    """
    Simulates NLU prediction using keyword matching.
    """
    
    # Normalize prompt for matching
    p = prompt.lower()
    
    # Check for specific intents based on the active domain
    if domain == "Travel & Booking":
        if "book" in p or "tickets" in p or "reservation" in p:
            intent = "book_flight" if "flight" in p or "plane" in p else "book_ticket"
            
            # Simple entity extraction simulation
            destination = "an unknown destination"
            if "to paris" in p:
                destination = "Paris"
            elif "to london" in p:
                destination = "London"
                
            entities = f"{{\"destination\":\"{destination}\", \"type\":\"{intent.split('_')[1]}\"}}"
            return intent, entities

    elif domain == "Finance":
        if "balance" in p or "how much" in p:
            return "query_balance", "{\"account\":\"checking\"}"
        elif "transfer" in p or "send money" in p:
            return "transfer_funds", "{\"amount\":\"unknown\"}"
            
    elif domain == "IT Support":
        if "password" in p or "reset" in p:
            return "reset_password", "{}"
        elif "login" in p or "troubleshoot" in p:
            return "troubleshoot_login", "{}"

    # General Intent checks (low confidence fallback in a real system)
    if "hello" in p or "hi" in p:
        return "greeting", "{}"
    if "trained the model" in p:
        # A specific response for the user's meta-query
        return "meta_query_training", "{}"

    # Default Fallback Intent
    return "default_fallback", "{}"

# The main chat handler, which now calls the new prediction stub
def handle_chat_input(workspace_name):
    domain = st.session_state.current_domain
    domain_data = DOMAINS.get(domain, {})
    domain_display = domain_data.get("icon", "") + " " + domain
    
    if prompt := st.chat_input(f"Chat with your '{workspace_name}' Bot..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.chat_history[workspace_name] = st.session_state.messages
        
        # --- NEW LOGIC: PREDICT & RESPOND ---
        intent, entities_json = predict_intent_and_entities(prompt, domain)
        entities_dict = json.loads(entities_json)
        
        response_lines = [
            f"**Prediction Successful!** (Simulated NLU)",
            "---",
            f"**Predicted Domain:** `{domain_display}`",
            f"**Predicted Intent:** `{intent}`",
            f"**Extracted Entities:** `{entities_dict}`"
        ]
        
        # Add a conditional response for better simulation
        if intent in ["book_ticket", "book_flight"]:
            response_lines.append(f"\n*Simulated Response:* Okay, I'm finding tickets to **{entities_dict.get('destination', 'your destination')}** now in the **{domain}** domain.")
        elif intent == "meta_query_training":
             response_lines.append(f"\n*Simulated Response:* That's great! My NLU component is ready. This chat window is now reflecting the *simulated* prediction results based on your trained domain.")
        elif intent == "default_fallback":
            response_lines.append(f"\n*Simulated Response:* I'm sorry, I don't know how to handle that request. Please try annotating more examples for the **{domain}** domain!")
        else:
             response_lines.append(f"\n*Simulated Response:* Got it! Proceeding with the **{intent}** action.")

        response = "\n".join(response_lines)
        # --- END NEW LOGIC ---
        
        with st.chat_message("assistant"):
            st.markdown(response)
            
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.session_state.chat_history[workspace_name] = st.session_state.messages
            
        st.rerun() 

# ==============================
# HOME PAGE / WORKSPACE MANAGER
# ==============================
def show_home_page():
    if not st.session_state.logged_in_email:
        navigate_to_login()
        st.rerun()
        return

    user_email_prefix = st.session_state.logged_in_email.split('@')[0]
    st.markdown(f"<div class='title'>Welcome, <span style='color:#6EC6FF;'>{user_email_prefix}!</span></div>", unsafe_allow_html=True)
    st.markdown("## Your BuddyBot Workspaces")
    st.markdown("---")
    
    user_email = st.session_state.logged_in_email
    local_cursor = conn.cursor()
    
    local_cursor.execute("SELECT workspace_name, domain, last_modified FROM workspaces WHERE user_email=?", (user_email,))
    existing_workspaces = local_cursor.fetchall()
    
    # 1. Determine the number of existing workspaces and the column index for the "Create New Project" card
    num_workspaces = len(existing_workspaces)
    
    # Calculate how many columns are needed (including the new project card)
    # Since we display 3 items per row, we calculate the total columns needed
    total_items = num_workspaces + 1
    
    # Create the columns
    cols = st.columns(3)
    
    # 2. Display Existing Workspaces
    if existing_workspaces:
        #st.markdown("### Activate Existing Bot or Create New")
        
        for i, (name, domain, modified) in enumerate(existing_workspaces):
            # Place existing workspaces into the appropriate column
            with cols[i % 3]:
                domain_icon = DOMAINS.get(domain, {}).get("icon", "💼")
                
                st.markdown(f"""
                <div class="domain-card">
                    <h3>{domain_icon} {name}</h3>
                    <p>Domain: {domain}</p>
                    <p>Last Activity: {modified[:10]}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Check if the current workspace is the active one to change button label/style
                is_current = st.session_state.current_workspace == name
                
                st.button(
                    "Active Bot" if is_current else "Activate Bot",
                    key=f"activate_ws_{name.replace(' ', '_')}",
                    use_container_width=True,
                    # Change type to 'secondary' if it is the current one
                    type="secondary" if is_current else "primary",
                    disabled=is_current,
                    on_click=activate_existing_workspace,
                    args=(name, domain)
                )
                st.markdown("<br>", unsafe_allow_html=True)
    else:
        st.markdown("### Create Your First BuddyBot Workspace")


    # 3. Display the "Create New Project" button in the next available column
    
    # Calculate the index for the 'Create New Project' card
    create_card_index = num_workspaces % 3
    
    with cols[create_card_index]:
        # Use custom HTML/CSS to create a card-like button
        st.markdown(f"""
        <div class="domain-card" style='border: 2px dashed #6EC6FF; height: 180px; display: flex; flex-direction: column; justify-content: center; align-items: center;'>
            <h3 style='color: #6EC6FF; margin-bottom: 5px; font-size: 36px;'>➕</h3>
            <p style='color: #6EC6FF; font-weight: bold; font-size: 16px;'>Create New Project</p>
            <p style='font-size: 12px; color: #ccc;'>Start a new NLU bot</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Streamlit button to handle the click action, placed directly under the custom card
        if st.button(
            "Go to Project Creation", 
            use_container_width=True, 
            key="go_to_create_btn_card", 
            type="primary"
        ):
            navigate_to_create_workspace()
            st.rerun()

    st.markdown("---")


# ==============================
# WORKSPACE CREATION PAGE
# ==============================
def show_create_workspace_page():
    st.markdown("<div class='title'>1. Name Your Workspace</div>", unsafe_allow_html=True)
    
    with st.form(key="workspace_name_form"):
        workspace_name_input = st.text_input(
            "Enter a name for your new Bot/Workspace (e.g., 'Finance-QBot', 'Sports-Stats')", 
            value=st.session_state.temp_workspace_name
        )
        st.session_state.temp_workspace_name = workspace_name_input 
        
        submitted = st.form_submit_button("Proceed to Select Domain", type="primary", use_container_width=True)
        
        if submitted and not st.session_state.temp_workspace_name.strip():
            st.error("Please enter a valid workspace name.")
            submitted = False

    if submitted:
        st.markdown("---")
        st.markdown("<div class='title'>2. Select Your Bot's Domain</div>", unsafe_allow_html=True)
        
        domain_names = list(DOMAINS.keys())
        
        # Display 3 domains per row
        for i in range(0, len(domain_names), 3):
            cols = st.columns(3)
            
            for j in range(3):
                if i + j < len(domain_names):
                    domain_name = domain_names[i + j]
                    data = DOMAINS[domain_name]
                    
                    with cols[j]:
                        st.markdown(f"""
                        <div class="domain-card">
                            <h3>{data['icon']} {domain_name}</h3>
                            <p>{data['description']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.button(
                            "Select Domain and Create",
                            key=f"select_domain_btn_{domain_name.lower().replace(' ', '_')}",
                            use_container_width=True,
                            type="primary",
                            on_click=finalize_workspace_creation, 
                            args=(st.session_state.temp_workspace_name, domain_name)
                        )
                        st.markdown("<br>", unsafe_allow_html=True)

    if st.button("← Back to Home", key="back_to_home_create"):
        navigate_to_home()
        st.rerun()

# ==============================
# ACTION CHOICE PAGE
# ==============================
def show_action_choice_page():
    if not st.session_state.current_workspace:
        navigate_to_home()
        st.rerun()
        return

    workspace_name = st.session_state.current_workspace
    domain = st.session_state.current_domain

    st.markdown(f"<div class='title'>What would you like to do with {workspace_name}?</div>", unsafe_allow_html=True)
    st.markdown(f"**Domain:** {domain}", unsafe_allow_html=True)
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4) 

    # Col 1: Chat/Test
    with col1:
        st.markdown("<div class='domain-card' style='height: 150px;'><h3>🚀 Test/Chat</h3><p>Chat with the current version of the bot.</p></div>", unsafe_allow_html=True)
        st.button("Go to Chat Interface", key="action_test", type="primary", use_container_width=True, on_click=set_workspace_action, args=("Test",))

    # Col 2: Upload/Train
    with col2:
        st.markdown("<div class='domain-card' style='height: 150px;'><h3>📚 Upload & Train</h3><p>Upload new data or trigger model training.</p></div>", unsafe_allow_html=True)
        st.button("Upload & Train Bot", key="action_train", type="primary", use_container_width=True, on_click=set_workspace_action, args=("Train",))
    
    # Col 3: ANNOTATE (NEW - Week 3)
    with col3:
        st.markdown("<div class='domain-card' style='height: 150px;'><h3>📝 Annotate Data</h3><p>Label intents and entities for model training.</p></div>", unsafe_allow_html=True)
        st.button("Start Annotation", key="action_annotate", type="primary", use_container_width=True, on_click=set_workspace_action, args=("Annotate",))

    # Col 4: Evaluate
    with col4:
        st.markdown("<div class='domain-card' style='height: 150px;'><h3>📈 Evaluate Model</h3><p>View performance metrics, accuracy, and test results.</p></div>", unsafe_allow_html=True)
        st.button("View Evaluation", key="action_evaluate", type="primary", use_container_width=True, on_click=set_workspace_action, args=("Evaluate",))

    st.markdown("---")
    if st.button("← Back to Workspaces Home", key="back_from_action_choice"):
        st.session_state.current_workspace = None
        st.session_state.current_domain = None
        navigate_to_home()
        st.rerun()

# ==============================
# ANNOTATION PAGE (FINAL CORRECTED VERSION)
# ==============================
def show_annotation_page():
    if not st.session_state.logged_in_email or not st.session_state.current_workspace:
        navigate_to_home()
        st.rerun()
        return

    user_email = st.session_state.logged_in_email
    workspace_name = st.session_state.current_workspace
    domain = st.session_state.current_domain 
    
    st.markdown(f"<div class='title'>📝 Annotate Data: <span style='color:#6EC6FF;'>{workspace_name}</span></div>", unsafe_allow_html=True)
    st.markdown(f"**Domain:** {DOMAINS.get(domain, {}).get('icon', '')} {domain}", unsafe_allow_html=True)
    st.markdown("---")
    
    # 1. DEBUG/LOAD THE DATASET
    if st.session_state.sentences_df is None:
        st.warning("Attempting to load dataset from database...")
        df = load_dataset_blob(user_email, workspace_name)
        
        if df is None or df.empty:
            st.error("Dataset not found in DB. Please go to **Upload & Train** to upload and *SAVE* a CSV first.")
            if st.button("Go to Train Page", key="go_to_train_from_annotate_fail"):
                set_workspace_action("Train")
            return
        
        # Split the loaded DataFrame into individual sentences/utterances
        st.session_state.sentences_df = split_dataframe_to_sentences(df)
        st.session_state.annotation_index = 0 
        
        if st.session_state.sentences_df.empty:
            st.error("The dataset was loaded but contains zero sentences after processing.")
            return

        st.success(f"Successfully loaded and split **{len(st.session_state.sentences_df)}** sentences!")
        st.rerun()


    # Continue with annotation process only if sentences_df is available
    sentences_df = st.session_state.sentences_df
    total_sentences = len(sentences_df)

    if total_sentences == 0:
        st.error("The uploaded CSV could not be processed into individual sentences/utterances (zero sentences found).")
        return

    # Check if annotation is complete
    if st.session_state.annotation_index >= total_sentences:
        st.balloons()
        st.success(f"🎉 Annotation Complete! You have labeled **{total_sentences}** sentences.")
        st.markdown("You can now go to **Upload & Train** to train your custom NLU model!")
        if st.button("Go to Train Data"):
            set_workspace_action("Train")
        st.markdown("---")
        if st.button("← Change Action", key="change_action_btn_annotate_done"):
            st.session_state.workspace_action = None
            navigate_to_action_choice()
            st.rerun()
        return

    # 2. Display the current sentence & Pre-load existing data
    current_index = st.session_state.annotation_index
    current_sentence = sentences_df.loc[current_index, 'sentence']
    
    # --- START PRE-POPULATION LOGIC (NEW/CORRECTED) ---
    # Fetch existing data from the database using the helper function
    existing_intent, existing_entities_json = get_existing_annotation(user_email, workspace_name, current_sentence)
    
    # Convert JSON entities back to the simple string format for the UI
    pre_populated_entities = json_to_simple_entities(existing_entities_json)
    
    # Prepare intent options
    intents = DOMAINS.get(domain, {}).get("intents", ["greeting", "inform", "request", "default"])
    intent_options_with_none = ["-- Select Intent --"] + intents
    
    # Determine the initial selection index
    initial_intent_value = existing_intent if existing_intent else "-- Select Intent --"
    initial_index = intent_options_with_none.index(initial_intent_value) if initial_intent_value in intent_options_with_none else 0
    # --- END PRE-POPULATION LOGIC ---

    st.progress(current_index / total_sentences, text=f"Progress: {current_index + 1}/{total_sentences} sentences to process.")
    
    st.markdown("### Sentence to Annotate:")
    st.markdown(f'<div class="sentence-display" id="sentence-to-annotate">{current_sentence}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Annotation Tools")

    # 3. Intent Tagging (Dropdown) - Uses initial_index for pre-population
    intent_key = f"intent_select_{current_index}"
    selected_intent = st.selectbox("1. Select Intent:", intent_options_with_none, index=initial_index, key=intent_key)

    # 4. Entity Span Tagging (Simulation) - Uses pre_populated_entities for pre-population
    st.markdown("2. Highlight Entities (Simulated):")
    st.info("💡 **Enter Entities:** Use the format `entity_name:value, another_entity:value`. E.g., `artist:Monet, date:1872`")
    entity_key = f"entity_input_{current_index}"
    entity_input = st.text_input("Enter Entities (name:value, name:value...)", value=pre_populated_entities, key=entity_key)

    # 5. Save Labeled Data
    col_prev, col_save, col_skip = st.columns([1, 2, 1])

    with col_prev:
        if st.button("← Previous", use_container_width=True, disabled=(current_index == 0), key="prev_btn"):
            st.session_state.annotation_index = max(0, current_index - 1)
            st.rerun()

    with col_save:
        if st.button("✅ Save & Next", use_container_width=True, type="primary", key="save_btn"):
            
            if selected_intent == "-- Select Intent --":
                st.error("Please select a valid Intent before saving.")
                return

            # --- VALIDATION/FORMATTING LOGIC ---
            entities_dict = {}
            valid_entity_format = True
            try:
                if entity_input.strip():
                    parts = entity_input.split(',')
                    for part in parts:
                        if ':' in part:
                            k, v = part.split(':', 1)
                            # Ensure keys and values are clean before storage
                            entities_dict[k.strip()] = v.strip() 
                        else:
                            raise ValueError("Entity format error")

            except ValueError:
                st.error("Error parsing entities. Ensure format is: `entity_name:value, another_entity:value`")
                valid_entity_format = False
            except Exception as e:
                st.error(f"An unexpected error occurred during entity parsing: {e}")
                valid_entity_format = False


            if valid_entity_format:
                entities_json = json.dumps(entities_dict)

                # --- NEW/FIXED: USE THE UPSERT HELPER FUNCTION ---
                if save_annotation_to_db(workspace_name, user_email, current_sentence, selected_intent, entities_json):
                    # Only advance index if save was successful
                    st.session_state.annotation_index += 1 
                    st.toast(f"Saved: Intent='{selected_intent}'", icon='📝')
                    st.rerun()
                # --- END FIXED LOGIC ---


    with col_skip:
        if st.button("→ Skip", use_container_width=True, key="skip_btn"):
            st.session_state.annotation_index += 1
            st.toast("Sentence skipped.", icon='⏭️')
            st.rerun()
            
    st.markdown("---")
    local_cursor = conn.cursor()
    local_cursor.execute("SELECT COUNT(*) FROM annotations WHERE workspace_name=?", (workspace_name,))
    total_labeled = local_cursor.fetchone()[0]
    st.info(f"**Total Labeled Examples Saved in DB:** {total_labeled}")
    
    if st.button("← Change Action", key="back_from_annotate"):
        st.session_state.workspace_action = None
        navigate_to_action_choice()
        st.rerun()
# ==============================
# WORKSPACE / CHAT PAGE (RESTRICTED BY ACTION)
# ==============================
def show_workspace_page():
    # --- START OF EXISTING SETUP ---
    if not st.session_state.logged_in_email or not st.session_state.current_workspace or not st.session_state.workspace_action:
        navigate_to_home() 
        st.rerun()
        return

    user_email = st.session_state.logged_in_email
    workspace_name = st.session_state.current_workspace
    domain = st.session_state.current_domain 
    action = st.session_state.workspace_action
    
    # Assuming DOMAINS is defined globally
    domain_data = DOMAINS.get(domain, {"icon": "", "description": ""})
    domain_display = domain_data["icon"] + " " + domain
    
    st.markdown(f"<div class='title'>🤖 Workspace: {workspace_name}</div>", unsafe_allow_html=True)
    st.markdown(f"**Domain:** {domain_display} | **Action:** **{action}**", unsafe_allow_html=True)
    
    if st.button("← Change Action", key="change_action_btn"):
        st.session_state.workspace_action = None
        navigate_to_action_choice()
        st.rerun()

    st.markdown("---")
    # --- END OF EXISTING SETUP ---
    
    if action == "Train":
        # --- TRAIN MODE: Show Uploader, Data Preview, and Model Training (Single Column) ---
        
        st.subheader("1. Upload/Prepare Data")
        
        local_cursor = conn.cursor()
        local_cursor.execute("SELECT filename FROM datasets WHERE user_email=? AND workspace_name=?", (user_email, workspace_name))
        existing_file = local_cursor.fetchone()
        dataset_is_saved = existing_file is not None # <--- New flag for conditional display
        
        if existing_file:
            st.info(f"Existing Dataset: **{existing_file[0]}** is saved. Uploading a new file will overwrite it.")

        file = st.file_uploader("Upload a CSV dataset", type=["csv"], key="dataset_uploader")
        
        if file is not None:
            try:
                df = pd.read_csv(file)
                st.subheader("Dataset Preview")
                st.dataframe(df.head())
                
                file_data_bytes = file.getvalue()
                
                if st.button(f"Save Data to Workspace", use_container_width=True, type="primary", key="save_data_btn"):
                    local_cursor_train = conn.cursor()
                    # Use a proper transaction for DELETE and INSERT
                    local_cursor_train.execute("DELETE FROM datasets WHERE user_email=? AND workspace_name=?", (user_email, workspace_name))
                    # Pass file_data_bytes directly, use sqlite3.Binary if needed but raw bytes often work
                    local_cursor_train.execute("""
                        INSERT INTO datasets (user_email, workspace_name, filename, data) 
                        VALUES (?, ?, ?, ?)
                    """, (user_email, workspace_name, file.name, Binary(file_data_bytes))) 
                    local_cursor_train.execute("UPDATE workspaces SET last_modified=CURRENT_TIMESTAMP WHERE user_email=? AND workspace_name=?", (user_email, workspace_name))
                    conn.commit()
                    
                    # CRITICAL: Invalidate sentence cache and reset index if new data is uploaded/saved
                    st.session_state.sentences_df = None 
                    st.session_state.annotation_index = 0
                    st.success(f"✅ Success! Data for **{workspace_name}** saved. Now **Annotate** or **Train**.")
                    st.rerun() # Rerun to refresh the success message and clear the file uploader
            except Exception as e:
                st.error(f"Error processing or saving dataset: {e}")


        # --- START OF MODIFIED SECTION 2 ---
        st.subheader("2. Train NLU Model")
        
        # Check for annotated data before allowing training
        # Assuming pd.read_sql is available
        annotated_data = pd.read_sql("SELECT * FROM annotations WHERE user_email=? AND workspace_name=?", conn, params=(user_email, workspace_name))
        annotation_count = len(annotated_data)
        
        if dataset_is_saved:
            if annotation_count > 0:
                st.info(f"Ready to train with **{annotation_count}** labeled examples.")
                
                # 1. Training Button (Visible if annotations exist)
                if st.button(f"Start Model Training", use_container_width=True, type="primary", key="train_model_btn"):
                    train_nlu_model(workspace_name, annotated_data)
                    
                st.markdown("<br>", unsafe_allow_html=True)
                # 2. Annotation Button (Visible if dataset is saved, even if training is possible)
                if st.button("Continue Annotation →", use_container_width=True, key="go_to_annotate_continue", type="secondary"):
                    set_workspace_action("Annotate")
                    st.rerun()

            else:
                # Dataset saved, but no annotations. Show the 'Go to Annotation' button prominently.
                st.warning("Dataset is ready! Please **Annotate Data** first to create labels for custom training.")
                if st.button("Go to Annotation Page →", use_container_width=True, type="primary", key="go_to_annotate_from_train"):
                    set_workspace_action("Annotate")
                    st.rerun()
        else:
            # Dataset not saved. Cannot annotate or train.
            st.warning("Please upload and **Save Data to Workspace** in step 1 before attempting to Annotate or Train.")
        # --- END OF MODIFIED SECTION 2 ---
            
    elif action == "Test":
        # --- TEST MODE: Show only Chat Interface ---
        st.subheader("Chat and Test Bot Response")
        with st.container(height=550):
            display_chat_messages()
        handle_chat_input(workspace_name)
        
    elif action == "Evaluate":
        # --- EVALUATE MODE: Show Metrics ---
        st.subheader("Bot Evaluation Metrics")
        
        local_cursor = conn.cursor()
        local_cursor.execute("SELECT model_engine, model_version, training_date FROM models WHERE workspace_name=?", (workspace_name,))
        model_meta = local_cursor.fetchone()

        if model_meta:
            st.info(f"**Current Model:** {model_meta[0]} ({model_meta[1]}) trained on {model_meta[2][:10]}")
            st.metric("Last Training Date", f"{model_meta[2][:10]}")
            st.metric("Test Accuracy (Simulated)", "85%", "4%")
            
            local_cursor.execute("SELECT COUNT(*) FROM annotations WHERE workspace_name=?", (workspace_name,))
            total_examples = local_cursor.fetchone()[0]
            st.metric("Total Labeled Examples", f"{total_examples}")

        else:
            st.warning("No model has been trained for this workspace yet. Use the **Upload & Train** page.")
            st.metric("Last Training Date", "N/A")
            st.metric("Test Accuracy (Simulated)", "N/A")

    else:
        st.error("Invalid action selected. Please navigate back and try again.")
    
    # --- Back Buttons (Keep existing back logic) ---
    st.markdown("---")
    if st.button("← Back to Workspaces Home", key="back_to_home_workspace"):
        st.session_state.current_workspace = None
        st.session_state.current_domain = None
        st.session_state.workspace_action = None
        st.session_state.messages = [] 
        navigate_to_home()
        st.rerun()

# Note: This requires 'conn', 'DOMAINS', 'navigate_to_home', 'navigate_to_action_choice', 'set_workspace_action', 
# 'train_nlu_model', 'display_chat_messages', and 'handle_chat_input' to be defined elsewhere in your script.

# ==============================
# AUTH & REGISTRATION PAGES
# ==============================
def show_register_page():
    st.markdown("<div class='title'>Welcome to Sign Up <span style='color:#6EC6FF;'>Buddy!</span></div>", unsafe_allow_html=True)

    with st.form(key="register_form"):
        name = st.text_input("Enter your name", key="reg_name")
        email = st.text_input("Enter your email", key="reg_email")
        password = st.text_input("Enter your password", type="password", key="reg_password")
        
        st.markdown(
            "By signing up, you agree to our <a href='?page=policy'>Terms and Privacy Policy</a>.", 
            unsafe_allow_html=True
        )
        agree = st.checkbox("I confirm I have read and agree to the policy.", key="register_agree_checkbox")

        if st.form_submit_button("Sign Up", type="primary", use_container_width=True):
            local_cursor = conn.cursor()
            if name and email and password and agree:
                hashed_pw = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
                try:
                    local_cursor.execute("INSERT INTO users (name, email, password) VALUES (?, ?, ?)", (name, email, hashed_pw))
                    conn.commit()
                    st.success("🎉 Registration successful! Please login.")
                    navigate_to_login()
                    st.rerun()
                except sqlite3.IntegrityError:
                    st.error("⚠️ Email already exists.")
            else:
                st.warning("⚠️ Fill all fields and confirm agreement to terms to continue.")

    st.markdown('<div style="text-align:center;">Already have an account?</div>', unsafe_allow_html=True)
    if st.button("Sign In", use_container_width=True, key="go_to_login_btn"):
        navigate_to_login()
        st.rerun()


def show_login_page():
    st.markdown("<div class='title'>Welcome Back, <span style='color:#6EC6FF;'>Buddy!</span></div>", unsafe_allow_html=True)

    with st.form(key="login_form"):
        email = st.text_input("Enter your email", key="log_email")
        password = st.text_input("Enter your password", type="password", key="log_password")

        if st.form_submit_button("Sign In", type="primary", use_container_width=True):
            local_cursor = conn.cursor()
            local_cursor.execute("SELECT password, email FROM users WHERE email=?", (email,))
            user_data = local_cursor.fetchone()
            
            if user_data and bcrypt.checkpw(password.encode('utf-8'), user_data[0]):
                st.success("✅ Login successful! Redirecting to Home...")
                st.session_state.logged_in_email = user_data[1] 
                navigate_to_home() 
                st.rerun()
            else:
                st.error("❌ Invalid email or password.")

    st.markdown('<div style="text-align:center;">Don\'t have an account?</div>', unsafe_allow_html=True)
    if st.button("Sign Up", use_container_width=True, key="go_to_register_btn"):
        navigate_to_register()
        st.rerun()

def show_policy_page(): 
    st.markdown("<div class='title'>Terms and Privacy</div>", unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("""
        <div style="padding: 0 20px; text-align: left;">
        <h2>1. Terms of Service</h2>
        <p>Welcome to Buddy! By using our service, you agree to these Terms. Please read them carefully.</p>
        <h4>Account Responsibility</h4>
        <p>You are responsible for all activity under your account. You must keep your password secure and notify us immediately of any unauthorized use.</p>
        <h4>User Conduct</h4>
        <p>You agree not to use the Service for any unlawful or prohibited activities, including the transmission of harassing, hateful, or abusive content.</p>
        <h2>2. Privacy Policy</h2>
        <p>We take your privacy seriously. This policy describes how we collect, use, and handle your information.</p>
        <h4>Data Collection</h4>
        <p>We collect personal information you provide directly to us, suchs as your name, email address, and interactions with the 'Buddy' AI.</p>
        <h4>Data Usage</h4>
        <p>We use your data to operate, maintain, and improve our services. We do not sell your personal data to third parties.</p>
        </div>
    """, unsafe_allow_html=True)

    if st.button("← Back to Registration", key="back_to_reg"):
        navigate_to_register()
        st.rerun()

# ==============================
# PAGE LAYOUT EXECUTION
# ==============================

# CALL SIDEBAR ONCE HERE - IT WILL RUN ON EVERY PAGE EXCEPT 'register'
show_sidebar_content()

if st.session_state.page == 'workspace':
    show_workspace_page()
elif st.session_state.page == 'annotate': 
    show_annotation_page()
elif st.session_state.page == 'action_choice':
    show_action_choice_page()
elif st.session_state.page == 'home':
    show_home_page()
elif st.session_state.page == 'create_workspace':
    show_create_workspace_page()
else:
    # Landing page for Register/Login/Policy
    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown('<div class="logo-container"><img src="https://cdn-icons-png.flaticon.com/512/4712/4712100.png" width="150"></div>', unsafe_allow_html=True)
        st.markdown("""
            <div class='chat-bubble-container'>
                <div class='chat-bubble'>Hello, can you help me?</div><br>
                <div class='chat-bubble'>Of course! Buddy is ready to assist.</div>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='main-card'>", unsafe_allow_html=True)
        if st.session_state.page == 'register':
            show_register_page()
        elif st.session_state.page == 'login':
            show_login_page()
        elif st.session_state.page == 'policy':
            show_policy_page() 
        st.markdown("</div>", unsafe_allow_html=True)